"""Diversity-hard selection: beauty-led multi-objective, quotas enforced."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from nomen.diversity.clustering import cluster_candidates, same_family
from nomen.diversity.features import cv_pattern, phonetic_root, soft_consonant_ratio, vowel_a_ratio
from nomen.diversity.novelty import NoveltyArchive
from nomen.models import Candidate, Stage
from nomen.scoring.overall import compute_overall

# Target length distribution
LENGTH_QUOTAS = {5: 0.15, 6: 0.25, 7: 0.30, 8: 0.20, 9: 0.10}

# Max share any single generator may take of final shortlist
MAX_GENERATOR_SHARE = 0.20

# Max share any phonetic root may take
MAX_ROOT_SHARE = 0.05

# Minimum novelty to survive (hard) — diversity constraint, not ranking king
MIN_NOVELTY = 50.0

# Minimum beauty to survive (hard) — brands, not synthetic identifiers
MIN_BEAUTY = 72.0

# Preferred CV pattern budget — no pattern > 25%
MAX_PATTERN_SHARE = 0.25


@dataclass
class SelectionResult:
    selected: list[Candidate] = field(default_factory=list)
    rejected: list[Candidate] = field(default_factory=list)
    converged: bool = False
    convergence_reason: str = ""
    stats: dict[str, float] = field(default_factory=dict)


class DiversitySelector:
    def __init__(
        self,
        archive: NoveltyArchive | None = None,
        workers: int | None = None,
    ) -> None:
        self.archive = archive or NoveltyArchive()
        self.workers = workers

    def score_novelty(
        self,
        candidates: list[Candidate],
        *,
        winners: list[str],
    ) -> None:
        names = [c.name for c in candidates]
        scores = self.archive.novelty_scores_batch(
            names, population=names, winners=winners, k=8, workers=self.workers
        )
        for cand, nov in zip(candidates, scores):
            soft = soft_consonant_ratio(cand.name)
            if soft > 0.45:
                nov -= (soft - 0.45) * 80
            if vowel_a_ratio(cand.name) > 0.28:
                nov -= (vowel_a_ratio(cand.name) - 0.28) * 70
            cand.scores.novelty_score = round(max(0.0, min(100.0, nov)), 2)
            brand = cand.scores.brand_score or 0.0
            cand.scores.brand_score = brand
            cand.scores.overall = compute_overall(
                beauty=cand.scores.beauty_score,
                brand=brand,
                novelty=cand.scores.novelty_score,
                collision=cand.scores.collision_score,
            )
            cand.scores.cv_pattern = cv_pattern(cand.name)
            cand.scores.phonetic_root = phonetic_root(cand.name)

    def select(
        self,
        candidates: list[Candidate],
        *,
        limit: int,
        winners: list[str],
        min_beauty: float = MIN_BEAUTY,
    ) -> SelectionResult:
        result = SelectionResult()
        if not candidates:
            return result

        self.score_novelty(candidates, winners=winners)

        # 1) Hard beauty floor — reject synthetic / un-premium names
        pool: list[Candidate] = []
        for c in candidates:
            if c.scores.beauty_score < min_beauty:
                result.rejected.append(
                    c.reject(
                        Stage.BEAUTY,
                        f"beauty {c.scores.beauty_score:.1f} < {min_beauty}",
                    )
                )
            elif c.scores.novelty_score < MIN_NOVELTY:
                result.rejected.append(
                    c.reject(
                        Stage.NOVELTY,
                        f"novelty {c.scores.novelty_score:.1f} < {MIN_NOVELTY}",
                    )
                )
            else:
                pool.append(c)

        # 2) Cluster → one representative each
        clusters = cluster_candidates(pool)
        reps = [cl.representative for cl in clusters]
        for cl in clusters:
            for m in cl.members:
                if m.name != cl.representative.name:
                    result.rejected.append(
                        m.reject(
                            Stage.DIVERSITY,
                            f"clustered under '{cl.representative.name}' "
                            f"(root={phonetic_root(m.name)})",
                        )
                    )

        # 3) Drop anything in same family as existing winners
        filtered: list[Candidate] = []
        for c in reps:
            conflict = next((w for w in winners if same_family(c.name, w)), None)
            if conflict:
                result.rejected.append(
                    c.reject(Stage.DIVERSITY, f"same family as winner '{conflict}'")
                )
            else:
                filtered.append(c)

        # 4) Greedy fill with hard quotas
        selected: list[Candidate] = []
        gen_counts: Counter[str] = Counter()
        root_counts: Counter[str] = Counter()
        len_counts: Counter[int] = Counter()
        pat_counts: Counter[str] = Counter()

        # Beauty-led ranking; novelty must not dominate
        filtered.sort(
            key=lambda c: (
                c.scores.overall,
                c.scores.beauty_score,
                c.scores.brand_score,
                c.scores.novelty_score,
            ),
            reverse=True,
        )

        def can_take(c: Candidate, n_sel: int) -> str | None:
            if n_sel <= 0:
                return None
            # Generator quota
            gshare = (gen_counts[c.generator] + 1) / max(n_sel + 1, 1)
            if gen_counts[c.generator] > 0 and gshare > MAX_GENERATOR_SHARE + 1e-9:
                # Allow until we have enough to measure; enforce after 5 picks
                if n_sel >= 5 and (gen_counts[c.generator] + 1) / (n_sel + 1) > MAX_GENERATOR_SHARE:
                    return f"generator '{c.generator}' exceeds {MAX_GENERATOR_SHARE:.0%}"
            # Root quota — at most 1 until large N, then 5%
            root = c.scores.phonetic_root or phonetic_root(c.name)
            max_root = max(1, int(limit * MAX_ROOT_SHARE))
            if root_counts[root] >= max_root:
                return f"root '{root}' already at cap {max_root}"
            # Same root as already selected
            if root_counts[root] >= 1 and limit <= 40:
                return f"root '{root}' already represented"
            # Length soft-hard: don't exceed target+slack
            L = len(c.name)
            target = LENGTH_QUOTAS.get(L, 0.1)
            if n_sel >= 8:
                if (len_counts[L] + 1) / (n_sel + 1) > target + 0.12:
                    return f"length {L} over-represented"
            # Pattern
            pat = c.scores.cv_pattern or cv_pattern(c.name)
            if n_sel >= 8 and (pat_counts[pat] + 1) / (n_sel + 1) > MAX_PATTERN_SHARE:
                return f"pattern {pat} over-represented"
            # Visual near any selected
            for s in selected:
                if same_family(c.name, s.name):
                    return f"visual/phonetic near '{s.name}'"
            return None

        # First pass
        deferred: list[Candidate] = []
        for c in filtered:
            if len(selected) >= limit:
                deferred.append(c)
                continue
            reason = can_take(c, len(selected))
            if reason:
                deferred.append(c)
                continue
            selected.append(c)
            gen_counts[c.generator] += 1
            root_counts[c.scores.phonetic_root or phonetic_root(c.name)] += 1
            len_counts[len(c.name)] += 1
            pat_counts[c.scores.cv_pattern or cv_pattern(c.name)] += 1

        # Second pass — relax length/pattern slightly to fill quota
        if len(selected) < limit:
            for c in deferred:
                if len(selected) >= limit:
                    result.rejected.append(
                        c.reject(Stage.DIVERSITY, "shortlist full after diversity pass")
                    )
                    continue
                root = c.scores.phonetic_root or phonetic_root(c.name)
                if root_counts[root] >= 1:
                    result.rejected.append(
                        c.reject(Stage.DIVERSITY, f"root '{root}' already represented")
                    )
                    continue
                if any(same_family(c.name, s.name) for s in selected):
                    result.rejected.append(
                        c.reject(Stage.DIVERSITY, "near selected candidate")
                    )
                    continue
                if gen_counts[c.generator] / max(len(selected) + 1, 1) > MAX_GENERATOR_SHARE and len(selected) >= 5:
                    result.rejected.append(
                        c.reject(Stage.DIVERSITY, f"generator quota '{c.generator}'")
                    )
                    continue
                selected.append(c)
                gen_counts[c.generator] += 1
                root_counts[root] += 1
                len_counts[len(c.name)] += 1
                pat_counts[c.scores.cv_pattern or cv_pattern(c.name)] += 1
            # leftover deferred
            for c in deferred:
                if c not in selected and c.rejected_at is None:
                    result.rejected.append(
                        c.reject(Stage.DIVERSITY, "not selected under diversity constraints")
                    )

        result.selected = selected

        # Convergence detection: if top roots dominate or soft letters explode
        if selected:
            top_root, top_n = root_counts.most_common(1)[0]
            soft_avg = sum(soft_consonant_ratio(c.name) for c in selected) / len(selected)
            gen_dom = max(gen_counts.values()) / len(selected) if selected else 0
            if top_n / len(selected) >= 0.25 and len(selected) >= 8:
                result.converged = True
                result.convergence_reason = (
                    f"root '{top_root}' occupies {top_n}/{len(selected)} of shortlist"
                )
            elif soft_avg >= 0.42 and len(selected) >= 8:
                result.converged = True
                result.convergence_reason = f"soft-letter convergence avg={soft_avg:.2f}"
            elif gen_dom >= 0.35 and len(selected) >= 8:
                result.converged = True
                result.convergence_reason = "single generator domination"

        result.stats = {
            "selected": float(len(selected)),
            "clusters": float(len(clusters)),
            "unique_roots": float(len(root_counts)),
            "unique_generators": float(len(gen_counts)),
            "unique_patterns": float(len(pat_counts)),
        }
        return result

    def detect_list_convergence(self, names: list[str]) -> tuple[bool, str]:
        if len(names) < 8:
            return False, ""
        roots = Counter(phonetic_root(n) for n in names)
        top_root, n = roots.most_common(1)[0]
        if n / len(names) >= 0.20:
            return True, f"winner archive root '{top_root}' share {n}/{len(names)}"
        # Pairwise family density
        hits = 0
        total = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                total += 1
                if same_family(names[i], names[j]):
                    hits += 1
        if total and hits / total >= 0.08:
            return True, f"pairwise family density {hits}/{total}"
        return False, ""
