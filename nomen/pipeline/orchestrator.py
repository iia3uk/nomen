"""Novelty-search hunt loop — diversity is a hard constraint."""

from __future__ import annotations

import asyncio
import sys
import time
from collections import Counter
from pathlib import Path

import diskcache
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from nomen.config import AppConfig, Secrets
from nomen.diversity.novelty import NoveltyArchive
from nomen.diversity.selector import DiversitySelector
from nomen.filters.offline import OfflineFilterBank
from nomen.generation.engine import GenerationEngine
from nomen.linguistics import normalize, reserved_brands
from nomen.models import Candidate, HuntState, Scores, Stage
from nomen.pipeline.checkpoint import Checkpoint
from nomen.pipeline.export import export_all
from nomen.plugins import iter_generators, iter_validators
from nomen.scoring.beauty import passes_beauty_gates
from nomen.scoring.elite import EliteArchive
from nomen.scoring.scorer import score_candidate
from nomen.scoring.tournament import run_tournament
from nomen.similarity.engine import SimilarityEngine
from nomen.validation.gateway import ValidationGateway

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console(legacy_windows=False, soft_wrap=True)


def _reason_tally(cands: list[Candidate], n: int = 4) -> str:
    cnt: Counter[str] = Counter()
    for c in cands:
        if c.rejected_at:
            cnt[c.rejected_at.value] += 1
        elif c.rejection_reasons:
            cnt[c.rejection_reasons[0].split(":", 1)[0]] += 1
    if not cnt:
        return ""
    return "  (" + ", ".join(f"{k} {v}" for k, v in cnt.most_common(n)) + ")"


def _stage(label: str, kept: int, dropped: int = 0, elapsed: float | None = None, extra: str = "") -> None:
    drop = f"  drop={dropped:,}" if dropped else ""
    timing = f"  {elapsed:.1f}s" if elapsed is not None else ""
    console.print(f"  [dim]→[/] {label}: [green]{kept:,}[/]{drop}{extra}{timing}")


def install_uvloop() -> None:
    if sys.platform == "win32":
        return
    try:
        import uvloop

        uvloop.install()
    except ImportError:
        pass


async def run_engine(cfg: AppConfig, secrets: Secrets) -> list[Candidate]:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = diskcache.Cache(str(out / "cache"))
    ckpt = Checkpoint(out / "checkpoint.json")

    state = ckpt.load(cfg.seed) if cfg.resume else None
    resumed = state is not None
    if state is None:
        state = HuntState(seed=cfg.seed)

    reserved = reserved_brands()
    if reserved and state.clean_names:
        dropped = [n for n in state.clean_names if normalize(n) in reserved]
        if dropped:
            console.print(
                "[yellow]Dropped reserved/owned brands from checkpoint:[/] "
                + ", ".join(n[:1].upper() + n[1:] for n in dropped)
            )
            keep = {n for n in state.clean_names if normalize(n) not in reserved}
            state.clean_names = [n for n in state.clean_names if n in keep]
            for n in dropped:
                state.candidate_meta.pop(n, None)
                state.candidate_meta.pop(normalize(n), None)

    if resumed:
        console.print(
            f"[cyan]Resuming[/] {len(state.clean_names)} clean name(s) from {ckpt.path.name}  "
            f"— pass [bold]--fresh[/] to start a new hunt"
        )
        if len(state.clean_names) >= cfg.target_clean:
            console.print(
                "[yellow]Previous hunt already hit the target — not generating new names.[/]"
            )

    from nomen.parallel import default_workers, parallel_map

    cpu_workers = cfg.workers if cfg.workers and cfg.workers > 0 else default_workers()
    console.print(
        f"[cyan]Parallelism:[/] gen_workers={cpu_workers}  "
        f"http_concurrency={cfg.concurrency}"
    )
    http_per_name = 10 + len(cfg.tlds) + (4 if secrets.has_web_search else 1)
    console.print(
        f"[dim]HTTP budget:[/] ~{http_per_name} req/name × online_batch={cfg.online_batch} "
        f"per round. 404 cache 6h / hits 72h. "
        f"GitHub search without token ≈ 10 req/min. "
        f"No USPTO/Rospatent API — trademark is search-proxy only."
    )

    engine = GenerationEngine(cfg.seed, cfg.min_len, cfg.max_len, workers=cpu_workers)
    engine.set_winners(state.clean_names)
    engine.set_archive(state.archive_names or state.clean_names)

    archive = NoveltyArchive()
    archive.extend(state.clean_names)
    archive.extend(state.archive_names)
    selector = DiversitySelector(archive, workers=cpu_workers)
    elite = EliteArchive(out / "elite_archive.json", capacity=64)
    min_beauty = cfg.min_beauty_score

    offline = OfflineFilterBank()
    similar = SimilarityEngine(
        cfg.similarity,
        embedding_model=cfg.embeddings.model if cfg.embeddings.enabled else None,
    )
    if cfg.embeddings.enabled:
        console.print("[cyan]Loading embedding model…[/]")
        similar.enable_embeddings(True)

    gateway = ValidationGateway(cfg, secrets, cache)
    checked: list[Candidate] = []
    rejected: list[Candidate] = []
    clean: list[Candidate] = []

    for name in state.clean_names:
        meta = state.candidate_meta.get(name, {})
        c = Candidate(name=name, generator=meta.get("generator", "resume"))
        if "scores" in meta:
            c.scores = Scores.model_validate(meta["scores"])
        c.mark_clean()
        clean.append(c)
        checked.append(c)

    try:
        round_i = state.round
        stagnant_rounds = 0
        while len(clean) < cfg.target_clean:
            if cfg.generation.rounds_max and round_i >= cfg.generation.rounds_max:
                break
            round_i += 1
            state.round = round_i

            # Convergence restart
            converged, reason = selector.detect_list_convergence([c.name for c in clean])
            if converged:
                stagnant_rounds += 1
            if stagnant_rounds >= 1 or (round_i > 1 and round_i % 3 == 0):
                # Periodic exploration jump even without formal convergence
                pass
            if converged:
                console.print(f"[red]CONVERGENCE DETECTED:[/] {reason}")
                console.print("[yellow]Restarting exploration in a new search region…[/]")
                engine.reseed_exploration(reason)
                state.exploration_restarts += 1
                stagnant_rounds = 0

            console.rule(
                f"[bold]Round {round_i} — clean {len(clean)}/{cfg.target_clean} "
                f"(restarts={state.exploration_restarts})"
            )
            if clean:
                console.print(
                    "[dim]Holding:[/] " + ", ".join(c.display_name for c in clean)
                )
            console.print(
                f"Generating novelty batch of {cfg.generate_batch:,} "
                f"(chunked across CPU workers)…"
            )
            gen_done = {"n": 0}
            t_gen = time.perf_counter()

            def _on_generator(label: str, produced: int, unique: int) -> None:
                gen_done["n"] += 1
                console.print(
                    f"  [cyan]gen[/] {label:18} +{produced:,}  "
                    f"unique={unique:,}  job {gen_done['n']}"
                )

            provenance = engine.generate_all(cfg.generate_batch, on_generator=_on_generator)
            for plugin in iter_generators():
                for name in plugin.generate(max(100, cfg.generate_batch // 20)):
                    provenance.setdefault(name, f"plugin:{plugin.name}")

            stats = engine.last_stats
            _stage(
                "generated",
                len(provenance),
                dropped=int(stats.get("culled", 0)),
                elapsed=time.perf_counter() - t_gen,
                extra=f"  unique-raw={int(stats.get('raw_unique', len(provenance))):,}",
            )

            state.generated_total += len(provenance)
            t0 = time.perf_counter()
            survivors, bad = offline.apply(provenance)
            rejected.extend(bad)
            for c in bad:
                state.rejected[c.name] = list(c.rejection_reasons)
            _stage("offline filters", len(survivors), len(bad), time.perf_counter() - t0, _reason_tally(bad))

            t0 = time.perf_counter()
            survivors, sim_bad = similar.apply(survivors)
            rejected.extend(sim_bad)
            for c in sim_bad:
                state.rejected[c.name] = list(c.rejection_reasons)
            _stage("similarity", len(survivors), len(sim_bad), time.perf_counter() - t0, _reason_tally(sim_bad))

            # Multi-objective scores (beauty + brand + collision; novelty later)
            lm = engine.lm
            brand_floor = max(70.0, cfg.min_overall_score - 15)

            def _brand(c: Candidate) -> Candidate:
                return score_candidate(c, lm)

            t0 = time.perf_counter()
            survivors = parallel_map(_brand, survivors, workers=cpu_workers, chunksize=64)
            scored: list[Candidate] = []
            beauty_bad = 0
            brand_bad = 0
            for c in survivors:
                ok, why = passes_beauty_gates(c.name, min_beauty=min_beauty)
                if not ok:
                    c.reject(Stage.BEAUTY, "; ".join(why[:3]))
                    rejected.append(c)
                    state.rejected[c.name] = list(c.rejection_reasons)
                    beauty_bad += 1
                elif c.scores.brand_score < brand_floor:
                    c.reject(Stage.SCORE, f"brand_score {c.scores.brand_score:.1f} too low")
                    rejected.append(c)
                    state.rejected[c.name] = list(c.rejection_reasons)
                    brand_bad += 1
                else:
                    scored.append(c)
            _stage(
                "beauty+brand",
                len(scored),
                beauty_bad + brand_bad,
                time.perf_counter() - t0,
                f"  (beauty {beauty_bad}, brand {brand_bad})",
            )

            # Diversity-hard selection for online queue
            need = max(cfg.online_batch * 2, cfg.target_clean * 6)
            sel = selector.select(
                scored,
                limit=need,
                winners=[c.name for c in clean],
                min_beauty=min_beauty,
            )
            rejected.extend(sel.rejected)
            for c in sel.rejected:
                state.rejected[c.name] = list(c.rejection_reasons)

            if sel.converged:
                console.print(f"[red]Batch convergence:[/] {sel.convergence_reason}")
                engine.reseed_exploration(sel.convergence_reason)
                state.exploration_restarts += 1

            # Human simulation: 100 virtual users, pairwise tournament
            tourney_keep = max(cfg.online_batch, cfg.target_clean * 3)
            tourney = run_tournament(
                sel.selected,
                keep=tourney_keep,
                n_users=100,
                seed=(sum(ord(ch) for ch in cfg.seed) * 131 + round_i) & 0xFFFFFFFF,
            )
            for c in tourney.eliminated:
                c.reject(Stage.TOURNAMENT, "lost pairwise human-simulation tournament")
                rejected.append(c)
                state.rejected[c.name] = list(c.rejection_reasons)

            queue = tourney.winners
            elite_drop = 0
            # Compete against permanent elite archive (soft gate when full)
            elite_floor = elite.must_beat_floor()
            if elite_floor > 0:
                kept_q: list[Candidate] = []
                for c in queue:
                    if c.scores.overall + 1.5 < elite_floor and c.scores.beauty_score < min_beauty + 3:
                        c.reject(
                            Stage.SCORE,
                            f"below elite floor overall={c.scores.overall:.1f} < {elite_floor:.1f}",
                        )
                        rejected.append(c)
                        state.rejected[c.name] = list(c.rejection_reasons)
                    else:
                        kept_q.append(c)
                elite_drop = len(queue) - len(kept_q)
                queue = kept_q

            console.print(
                f"  [dim]→[/] diversity/tournament: [green]{len(queue)}[/] online queue  "
                f"(roots={sel.stats.get('unique_roots', 0):.0f}, "
                f"gens={sel.stats.get('unique_generators', 0):.0f}, "
                f"patterns={sel.stats.get('unique_patterns', 0):.0f})  "
                f"drop={len(sel.rejected) + len(tourney.eliminated) + elite_drop:,}"
            )
            console.print(
                f"  [dim]lifetime[/] generated={state.generated_total:,}  "
                f"checked={state.checked_total}  clean={len(clean)}/{cfg.target_clean}"
            )

            # Feed archive
            archive.extend(c.name for c in queue)
            engine.set_archive(list(dict.fromkeys(engine.archive + [c.name for c in queue]))[-800:])
            state.archive_names = engine.archive[-500:]

            if not queue:
                console.print("[yellow]Empty diverse queue — reseeding…[/]")
                engine.reseed_exploration("empty queue")
                state.exploration_restarts += 1
                ckpt.save(state)
                continue

            from nomen.diversity.clustering import same_family

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=False,
            ) as progress:
                task = progress.add_task("Online validation", total=len(queue))
                # Fire all validations concurrently; cancel remainder when target hit
                tasks = {
                    asyncio.create_task(gateway.validate(c)): c for c in queue
                }
                try:
                    while tasks and len(clean) < cfg.target_clean:
                        done, _pending = await asyncio.wait(
                            tasks.keys(), return_when=asyncio.FIRST_COMPLETED
                        )
                        for fut in done:
                            tasks.pop(fut, None)
                            try:
                                cand = fut.result()
                            except Exception:
                                progress.advance(task)
                                continue
                            for plugin in iter_validators():
                                if cand.clean:
                                    extra = await plugin.validate(cand)
                                    for reason in extra:
                                        cand.reject(
                                            Stage.REGISTRY,
                                            f"plugin:{plugin.name}: {reason}",
                                        )
                            state.checked_total += 1
                            state.candidate_meta[cand.name] = {
                                "generator": cand.generator,
                                "scores": cand.scores.model_dump(),
                            }
                            if cand.clean:
                                twin = next(
                                    (w.name for w in clean if same_family(cand.name, w.name)),
                                    None,
                                )
                                if twin:
                                    cand.reject(
                                        Stage.DIVERSITY, f"same family as clean '{twin}'"
                                    )
                            checked.append(cand)
                            if cand.clean:
                                clean.append(cand)
                                state.clean_names.append(cand.name)
                                archive.add(cand.name)
                                if elite.consider(cand):
                                    elite.save()
                                console.print(
                                    f"  [bold green]CLEAN[/] {cand.display_name}  "
                                    f"beauty={cand.scores.beauty_score:.1f} "
                                    f"brand={cand.scores.brand_score:.1f} "
                                    f"nov={cand.scores.novelty_score:.1f} "
                                    f"all={cand.scores.overall:.1f} "
                                    f"root={cand.scores.phonetic_root} "
                                    f"({cand.generator})"
                                    + (
                                        f"  [yellow]unverified={','.join(cand.meta.get('unverified') or [])}[/]"
                                        if cand.meta.get("unverified")
                                        else ""
                                    )
                                )
                            else:
                                rejected.append(cand)
                                state.rejected[cand.name] = list(cand.rejection_reasons)
                                why = cand.rejection_reasons[-1] if cand.rejection_reasons else "rejected"
                                console.print(f"  [dim]reject[/] {cand.display_name:12} {why}")
                            progress.advance(task)
                            progress.update(
                                task,
                                description=f"Online validation (clean={len(clean)}/{cfg.target_clean})",
                            )
                        if len(clean) >= cfg.target_clean:
                            break
                        if cfg.pause_seconds:
                            await asyncio.sleep(cfg.pause_seconds)
                finally:
                    for fut in tasks:
                        fut.cancel()
                    if tasks:
                        await asyncio.gather(*tasks.keys(), return_exceptions=True)
                engine.set_winners([c.name for c in clean])
                ckpt.save(state)

            # Final diversity audit of clean set
            bad_conv, why = selector.detect_list_convergence([c.name for c in clean])
            if bad_conv and len(clean) < cfg.target_clean:
                console.print(f"[red]Clean-set convergence:[/] {why} — purging cluster tails")
                # Keep first occurrence per family only
                from nomen.diversity.clustering import same_family

                kept: list[Candidate] = []
                for c in clean:
                    if any(same_family(c.name, k.name) for k in kept):
                        c.reject(Stage.DIVERSITY, "purged from converged clean set")
                        rejected.append(c)
                        if c.name in state.clean_names:
                            state.clean_names.remove(c.name)
                    else:
                        kept.append(c)
                clean = kept
                engine.reseed_exploration(why)
                state.exploration_restarts += 1

            ckpt.save(state)
            export_all(out, checked, rejected)
            console.print(f"Progress: clean=[green]{len(clean)}[/]/{cfg.target_clean}")

    finally:
        await gateway.close()
        ckpt.save(state)
        cache.close()

    clean = sorted(
        clean,
        key=lambda c: (
            c.scores.overall,
            c.scores.beauty_score,
            c.scores.brand_score,
        ),
        reverse=True,
    )
    elite.consider_many(clean)

    export_all(out, checked, rejected)
    console.rule("[bold green]Complete")
    console.print(f"Clean names: {len(clean)}/{cfg.target_clean}")
    console.print(f"Elite archive: {len(elite.entries)} @ {elite.path}")
    console.print(f"Exploration restarts: {state.exploration_restarts}")
    console.print(f"Output: {out.resolve()}")
    roots = {c.scores.phonetic_root for c in clean}
    console.print(f"Unique phonetic roots: {len(roots)} / {len(clean)}")
    for c in clean[:20]:
        console.print(
            f"  • {c.display_name:12} beauty={c.scores.beauty_score:5.1f} "
            f"brand={c.scores.brand_score:5.1f} nov={c.scores.novelty_score:5.1f} "
            f"all={c.scores.overall:5.1f} root={c.scores.phonetic_root:6} "
            f"({c.generator})"
        )
    return clean
