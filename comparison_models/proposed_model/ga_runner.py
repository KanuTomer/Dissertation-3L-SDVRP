# comparison_models/proposed_model/ga_runner.py

import time
import random
import json
import copy
from math import sqrt

from comparison_models.common.algorithms.crossover import order_crossover
from comparison_models.proposed_model.mutation import (
    swap_mutation,
    hybrid_mutation,
)
from comparison_models.common.algorithms.selection import tournament_select
from comparison_models.common.loaders.route_evaluator import load_merged, evaluate_route, split_depot_and_customers
from comparison_models.proposed_model.config import (
    USE_PACKING,
    PENALTY_ALPHA,
    USE_ENHANCED_MUTATION,
    MODEL_NAME,
    ENABLE_ADAPTIVE_DECODING,
    ENABLE_TINY_ROUTE_REPAIR,
    ROUTE_COUNT_PENALTY,
    SMALL_ROUTE_PENALTY,
    BALANCE_PENALTY_WEIGHT,
    ELITE_OVERFLOW_RATIO,
    TINY_ROUTE_CUSTOMER_THRESHOLD,
    TINY_ROUTE_FILL_THRESHOLD,
    ENABLE_CUSTOMER_RELOCATION_REPAIR,
    ENABLE_ROUTE_BALANCE_MUTATION,
    ENABLE_FINAL_BEST_REFINEMENT,
    MAX_RELOCATION_ATTEMPTS,
    MAX_DISTANCE_INCREASE_RATIO,
    OVERFLOW_ROUTE_PENALTY,
    MERGED_ROUTE_REWARD,
    FILL_BALANCE_WEIGHT,
    FILL_REWARD_WEIGHT,
    MEDIUM_DATASET_THRESHOLD,
    MEDIUM_DATASET_BALANCE_SCALE,
    MEDIUM_DATASET_FILL_BALANCE_WEIGHT,
    MEDIUM_DATASET_OVERFLOW_SCALE,
    MEDIUM_DATASET_ROUTE_COUNT_SCALE,
    MEDIUM_DATASET_FILL_REWARD_SCALE,
    MEDIUM_DATASET_SMALL_ROUTE_SCALE,
    LARGE_DATASET_THRESHOLD,
    LARGE_DATASET_BALANCE_SCALE,
    LARGE_DATASET_FILL_BALANCE_WEIGHT,
    LARGE_DATASET_OVERFLOW_SCALE,
    LARGE_DATASET_ROUTE_COUNT_SCALE,
    LARGE_DATASET_FILL_REWARD_SCALE,
    VERY_LARGE_DATASET_THRESHOLD,
    VERY_LARGE_DATASET_BALANCE_SCALE,
    VERY_LARGE_DATASET_FILL_BALANCE_WEIGHT,
    VERY_LARGE_DATASET_OVERFLOW_SCALE,
    VERY_LARGE_DATASET_ROUTE_COUNT_SCALE,
    VERY_LARGE_DATASET_FILL_REWARD_SCALE,
    SMALL_DATASET_ROUTE_BALANCE_MUTATION_PROBABILITY,
    MEDIUM_DATASET_ROUTE_BALANCE_MUTATION_PROBABILITY,
    LARGE_DATASET_ROUTE_BALANCE_MUTATION_PROBABILITY,
    VERY_LARGE_DATASET_ROUTE_BALANCE_MUTATION_PROBABILITY,
    SMALL_DATASET_RELOCATION_DISTANCE_INCREASE_RATIO,
    MEDIUM_DATASET_RELOCATION_DISTANCE_INCREASE_RATIO,
    LARGE_DATASET_RELOCATION_DISTANCE_INCREASE_RATIO,
    VERY_LARGE_DATASET_RELOCATION_DISTANCE_INCREASE_RATIO,
    SMALL_DATASET_FINAL_REFINEMENT_DISTANCE_RATIO,
    MEDIUM_DATASET_FINAL_REFINEMENT_DISTANCE_RATIO,
    LARGE_DATASET_FINAL_REFINEMENT_DISTANCE_RATIO,
    VERY_LARGE_DATASET_FINAL_REFINEMENT_DISTANCE_RATIO,
)


class GARunner:
    def __init__(self, config: dict):
        self.config = config or {}
        self.pop_size = int(self.config.get("pop_size", self.config.get("population_size", 80)))
        self.gens = int(self.config.get("gens", self.config.get("num_generations", 200)))
        self.cx_prob = float(self.config.get("cx_prob", self.config.get("crossover_prob", 0.9)))
        self.mut_prob = float(self.config.get("mut_prob", self.config.get("mutation_prob", 0.2)))
        self.max_boxes_per_route = int(self.config.get("max_boxes_per_route", self.config.get("maxboxes", 48)))
        self.seed = int(self.config.get("seed", 42))
        self.verbose = bool(self.config.get("verbose", False))
        self.progress_label = str(self.config.get("progress_label", MODEL_NAME))
        self.route_eval_cache: dict[tuple[int, ...], dict] = {}
        self.permutation_eval_cache: dict[tuple[str, tuple[int, ...]], tuple[float, dict]] = {}
        self.split_candidate_cache: dict[tuple[int, ...], list[dict]] = {}
        self.split_offsets = list(self.config.get("split_offsets", [-5, 0, 5]))
        self.very_large_split_offsets = list(self.config.get("very_large_split_offsets", [-5, 0, 2]))
        self.adaptive_top_fraction = float(self.config.get("adaptive_top_fraction", 0.35))
        self.adaptive_top_min = int(self.config.get("adaptive_top_min", 12))
        self.adaptive_every_generations = int(self.config.get("adaptive_every_generations", 1))
        self.enable_adaptive_decoding = bool(
            self.config.get("enable_adaptive_decoding", ENABLE_ADAPTIVE_DECODING)
        )
        self.enable_tiny_route_repair = bool(
            self.config.get("enable_tiny_route_repair", ENABLE_TINY_ROUTE_REPAIR)
        )
        self.route_count_penalty = float(self.config.get("route_count_penalty", ROUTE_COUNT_PENALTY))
        self.small_route_penalty = float(self.config.get("small_route_penalty", SMALL_ROUTE_PENALTY))
        self.balance_penalty_weight = float(self.config.get("balance_penalty_weight", BALANCE_PENALTY_WEIGHT))
        self.elite_overflow_ratio = float(self.config.get("elite_overflow_ratio", ELITE_OVERFLOW_RATIO))
        self.tiny_route_customer_threshold = int(self.config.get("tiny_route_customer_threshold", TINY_ROUTE_CUSTOMER_THRESHOLD))
        self.tiny_route_fill_threshold = float(self.config.get("tiny_route_fill_threshold", TINY_ROUTE_FILL_THRESHOLD))
        self.enable_customer_relocation_repair = bool(
            self.config.get("enable_customer_relocation_repair", ENABLE_CUSTOMER_RELOCATION_REPAIR)
        )
        self.enable_route_balance_mutation = bool(
            self.config.get("enable_route_balance_mutation", ENABLE_ROUTE_BALANCE_MUTATION)
        )
        self.enable_final_best_refinement = bool(
            self.config.get("enable_final_best_refinement", ENABLE_FINAL_BEST_REFINEMENT)
        )
        self.max_relocation_attempts = int(
            self.config.get("max_relocation_attempts", MAX_RELOCATION_ATTEMPTS)
        )
        self.max_distance_increase_ratio = float(
            self.config.get("max_distance_increase_ratio", MAX_DISTANCE_INCREASE_RATIO)
        )
        self.overflow_route_penalty = float(
            self.config.get("overflow_route_penalty", OVERFLOW_ROUTE_PENALTY)
        )
        self.merged_route_reward = float(
            self.config.get("merged_route_reward", MERGED_ROUTE_REWARD)
        )
        self.fill_balance_weight = float(
            self.config.get("fill_balance_weight", FILL_BALANCE_WEIGHT)
        )
        self.fill_reward_weight = float(
            self.config.get("fill_reward_weight", FILL_REWARD_WEIGHT)
        )
        self.medium_dataset_threshold = int(
            self.config.get("medium_dataset_threshold", MEDIUM_DATASET_THRESHOLD)
        )
        self.medium_dataset_balance_scale = float(
            self.config.get("medium_dataset_balance_scale", MEDIUM_DATASET_BALANCE_SCALE)
        )
        self.medium_dataset_fill_balance_weight = float(
            self.config.get("medium_dataset_fill_balance_weight", MEDIUM_DATASET_FILL_BALANCE_WEIGHT)
        )
        self.medium_dataset_overflow_scale = float(
            self.config.get("medium_dataset_overflow_scale", MEDIUM_DATASET_OVERFLOW_SCALE)
        )
        self.medium_dataset_route_count_scale = float(
            self.config.get("medium_dataset_route_count_scale", MEDIUM_DATASET_ROUTE_COUNT_SCALE)
        )
        self.medium_dataset_fill_reward_scale = float(
            self.config.get("medium_dataset_fill_reward_scale", MEDIUM_DATASET_FILL_REWARD_SCALE)
        )
        self.medium_dataset_small_route_scale = float(
            self.config.get("medium_dataset_small_route_scale", MEDIUM_DATASET_SMALL_ROUTE_SCALE)
        )
        self.large_dataset_threshold = int(
            self.config.get("large_dataset_threshold", LARGE_DATASET_THRESHOLD)
        )
        self.large_dataset_balance_scale = float(
            self.config.get("large_dataset_balance_scale", LARGE_DATASET_BALANCE_SCALE)
        )
        self.large_dataset_fill_balance_weight = float(
            self.config.get("large_dataset_fill_balance_weight", LARGE_DATASET_FILL_BALANCE_WEIGHT)
        )
        self.large_dataset_overflow_scale = float(
            self.config.get("large_dataset_overflow_scale", LARGE_DATASET_OVERFLOW_SCALE)
        )
        self.large_dataset_route_count_scale = float(
            self.config.get("large_dataset_route_count_scale", LARGE_DATASET_ROUTE_COUNT_SCALE)
        )
        self.large_dataset_fill_reward_scale = float(
            self.config.get("large_dataset_fill_reward_scale", LARGE_DATASET_FILL_REWARD_SCALE)
        )
        self.very_large_dataset_threshold = int(
            self.config.get("very_large_dataset_threshold", VERY_LARGE_DATASET_THRESHOLD)
        )
        self.very_large_dataset_balance_scale = float(
            self.config.get("very_large_dataset_balance_scale", VERY_LARGE_DATASET_BALANCE_SCALE)
        )
        self.very_large_dataset_fill_balance_weight = float(
            self.config.get("very_large_dataset_fill_balance_weight", VERY_LARGE_DATASET_FILL_BALANCE_WEIGHT)
        )
        self.very_large_dataset_overflow_scale = float(
            self.config.get("very_large_dataset_overflow_scale", VERY_LARGE_DATASET_OVERFLOW_SCALE)
        )
        self.very_large_dataset_route_count_scale = float(
            self.config.get("very_large_dataset_route_count_scale", VERY_LARGE_DATASET_ROUTE_COUNT_SCALE)
        )
        self.very_large_dataset_fill_reward_scale = float(
            self.config.get("very_large_dataset_fill_reward_scale", VERY_LARGE_DATASET_FILL_REWARD_SCALE)
        )
        self.small_dataset_route_balance_mutation_probability = float(
            self.config.get(
                "small_dataset_route_balance_mutation_probability",
                SMALL_DATASET_ROUTE_BALANCE_MUTATION_PROBABILITY,
            )
        )
        self.medium_dataset_route_balance_mutation_probability = float(
            self.config.get(
                "medium_dataset_route_balance_mutation_probability",
                MEDIUM_DATASET_ROUTE_BALANCE_MUTATION_PROBABILITY,
            )
        )
        self.large_dataset_route_balance_mutation_probability = float(
            self.config.get(
                "large_dataset_route_balance_mutation_probability",
                LARGE_DATASET_ROUTE_BALANCE_MUTATION_PROBABILITY,
            )
        )
        self.very_large_dataset_route_balance_mutation_probability = float(
            self.config.get(
                "very_large_dataset_route_balance_mutation_probability",
                VERY_LARGE_DATASET_ROUTE_BALANCE_MUTATION_PROBABILITY,
            )
        )
        self.small_dataset_relocation_distance_increase_ratio = float(
            self.config.get(
                "small_dataset_relocation_distance_increase_ratio",
                SMALL_DATASET_RELOCATION_DISTANCE_INCREASE_RATIO,
            )
        )
        self.medium_dataset_relocation_distance_increase_ratio = float(
            self.config.get(
                "medium_dataset_relocation_distance_increase_ratio",
                MEDIUM_DATASET_RELOCATION_DISTANCE_INCREASE_RATIO,
            )
        )
        self.large_dataset_relocation_distance_increase_ratio = float(
            self.config.get(
                "large_dataset_relocation_distance_increase_ratio",
                LARGE_DATASET_RELOCATION_DISTANCE_INCREASE_RATIO,
            )
        )
        self.very_large_dataset_relocation_distance_increase_ratio = float(
            self.config.get(
                "very_large_dataset_relocation_distance_increase_ratio",
                VERY_LARGE_DATASET_RELOCATION_DISTANCE_INCREASE_RATIO,
            )
        )
        self.small_dataset_final_refinement_distance_ratio = float(
            self.config.get(
                "small_dataset_final_refinement_distance_ratio",
                SMALL_DATASET_FINAL_REFINEMENT_DISTANCE_RATIO,
            )
        )
        self.medium_dataset_final_refinement_distance_ratio = float(
            self.config.get(
                "medium_dataset_final_refinement_distance_ratio",
                MEDIUM_DATASET_FINAL_REFINEMENT_DISTANCE_RATIO,
            )
        )
        self.large_dataset_final_refinement_distance_ratio = float(
            self.config.get(
                "large_dataset_final_refinement_distance_ratio",
                LARGE_DATASET_FINAL_REFINEMENT_DISTANCE_RATIO,
            )
        )
        self.very_large_dataset_final_refinement_distance_ratio = float(
            self.config.get(
                "very_large_dataset_final_refinement_distance_ratio",
                VERY_LARGE_DATASET_FINAL_REFINEMENT_DISTANCE_RATIO,
            )
        )
        self.max_full_candidate_evaluations = int(
            self.config.get("max_full_candidate_evaluations", 4)
        )
        self.final_refinement_split_offset = int(
            self.config.get("final_refinement_split_offset", 8)
        )
        self.final_refinement_overflow_ratio = float(
            self.config.get("final_refinement_overflow_ratio", 0.14)
        )
        self.num_customers = int(self.config.get("num_customers", 0))

    def is_large_dataset(self):
        return self.num_customers >= self.large_dataset_threshold

    def is_very_large_dataset(self):
        return self.num_customers >= self.very_large_dataset_threshold

    def dataset_scale(self):
        if self.is_very_large_dataset():
            return "very_large"
        if self.is_large_dataset():
            return "large"
        if self.num_customers >= self.medium_dataset_threshold:
            return "medium"
        return "small"

    def effective_policy(self):
        scale = self.dataset_scale()
        if scale == "very_large":
            return {
                "balance_penalty_weight": self.balance_penalty_weight * self.very_large_dataset_balance_scale,
                "fill_balance_weight": self.very_large_dataset_fill_balance_weight,
                "overflow_route_penalty": self.overflow_route_penalty * self.very_large_dataset_overflow_scale,
                "route_count_penalty": self.route_count_penalty * self.very_large_dataset_route_count_scale,
                "fill_reward_weight": self.fill_reward_weight * self.very_large_dataset_fill_reward_scale,
                "small_route_penalty": self.small_route_penalty,
                "route_balance_mutation_probability": self.very_large_dataset_route_balance_mutation_probability,
                "relocation_distance_increase_ratio": self.very_large_dataset_relocation_distance_increase_ratio,
                "final_refinement_distance_ratio": self.very_large_dataset_final_refinement_distance_ratio,
            }
        if scale == "large":
            return {
                "balance_penalty_weight": self.balance_penalty_weight * self.large_dataset_balance_scale,
                "fill_balance_weight": self.large_dataset_fill_balance_weight,
                "overflow_route_penalty": self.overflow_route_penalty * self.large_dataset_overflow_scale,
                "route_count_penalty": self.route_count_penalty * self.large_dataset_route_count_scale,
                "fill_reward_weight": self.fill_reward_weight * self.large_dataset_fill_reward_scale,
                "small_route_penalty": self.small_route_penalty,
                "route_balance_mutation_probability": self.large_dataset_route_balance_mutation_probability,
                "relocation_distance_increase_ratio": self.large_dataset_relocation_distance_increase_ratio,
                "final_refinement_distance_ratio": self.large_dataset_final_refinement_distance_ratio,
            }
        if scale == "medium":
            return {
                "balance_penalty_weight": self.balance_penalty_weight * self.medium_dataset_balance_scale,
                "fill_balance_weight": self.medium_dataset_fill_balance_weight,
                "overflow_route_penalty": self.overflow_route_penalty * self.medium_dataset_overflow_scale,
                "route_count_penalty": self.route_count_penalty * self.medium_dataset_route_count_scale,
                "fill_reward_weight": self.fill_reward_weight * self.medium_dataset_fill_reward_scale,
                "small_route_penalty": self.small_route_penalty * self.medium_dataset_small_route_scale,
                "route_balance_mutation_probability": self.medium_dataset_route_balance_mutation_probability,
                "relocation_distance_increase_ratio": self.medium_dataset_relocation_distance_increase_ratio,
                "final_refinement_distance_ratio": self.medium_dataset_final_refinement_distance_ratio,
            }
        return {
            "balance_penalty_weight": self.balance_penalty_weight,
            "fill_balance_weight": self.fill_balance_weight,
            "overflow_route_penalty": self.overflow_route_penalty,
            "route_count_penalty": self.route_count_penalty,
            "fill_reward_weight": self.fill_reward_weight,
            "small_route_penalty": self.small_route_penalty,
            "route_balance_mutation_probability": self.small_dataset_route_balance_mutation_probability,
            "relocation_distance_increase_ratio": self.small_dataset_relocation_distance_increase_ratio,
            "final_refinement_distance_ratio": self.small_dataset_final_refinement_distance_ratio,
        }

    def prescreen_route_count_slack(self):
        scale = self.dataset_scale()
        if scale == "very_large":
            return 3
        if scale == "large":
            return 2
        if scale == "medium":
            return 1
        return 1

    def route_drop_distance_ratio(self):
        scale = self.dataset_scale()
        if scale == "very_large":
            return 1.0005
        if scale == "large":
            return 1.001
        if scale == "medium":
            return 1.0005
        return 1.01

    @staticmethod
    def euclid(a, b):
        return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    @staticmethod
    def route_distance(depot, customers_map, route):
        if not route:
            return 0.0

        dist = 0.0

        first = customers_map[route[0]]
        dist += GARunner.euclid((depot[0], depot[1]), (first["x"], first["y"]))

        for i in range(len(route) - 1):
            c1 = customers_map[route[i]]
            c2 = customers_map[route[i + 1]]
            dist += GARunner.euclid((c1["x"], c1["y"]), (c2["x"], c2["y"]))

        last = customers_map[route[-1]]
        dist += GARunner.euclid((last["x"], last["y"]), (depot[0], depot[1]))

        return dist

    @staticmethod
    def build_customer_boxcount_map(customers):
        return {
            c["customer_id"]: len(c.get("assigned_boxes", []))
            for c in customers
        }

    @staticmethod
    def decode_by_boxcount(order, cust_boxcount_map, max_boxes_per_route=48):
        routes = []
        cur = []
        cur_boxes = 0

        for cid in order:
            b = cust_boxcount_map.get(cid, 0)

            if cur and (cur_boxes + b > max_boxes_per_route):
                routes.append(cur)
                cur = [cid]
                cur_boxes = b
            else:
                cur.append(cid)
                cur_boxes += b

        if cur:
            routes.append(cur)

        return routes

    def decode_with_adaptive_splits(
        self,
        order,
        cust_boxcount_map,
        depot,
        customers_map,
        max_boxes_per_route=48,
        distance_limit=None,
    ):
        routes = []
        cur = []
        cur_boxes = 0

        for cid in order:
            customer_boxes = cust_boxcount_map.get(cid, 0)
            should_split = False

            if cur and (cur_boxes + customer_boxes > max_boxes_per_route):
                should_split = True
            elif cur and distance_limit is not None:
                tentative_route = cur + [cid]
                tentative_distance = self.route_distance(depot, customers_map, tentative_route)
                enough_boxes_to_split = cur_boxes >= max(1, int(max_boxes_per_route * 0.5))
                route_is_long = tentative_distance > distance_limit
                if enough_boxes_to_split and route_is_long:
                    should_split = True

            if should_split:
                routes.append(cur)
                cur = [cid]
                cur_boxes = customer_boxes
            else:
                cur.append(cid)
                cur_boxes += customer_boxes

        if cur:
            routes.append(cur)

        return routes

    def route_box_count(self, route, cust_boxcount_map):
        return sum(cust_boxcount_map.get(customer_id, 0) for customer_id in route)

    def route_box_counts(self, routes, cust_boxcount_map):
        return [self.route_box_count(route, cust_boxcount_map) for route in routes]

    def estimate_partition_metrics(self, routes, cust_box_map, split_limit, overflow_ratio=None):
        route_box_counts = self.route_box_counts(routes, cust_box_map)
        route_customer_counts = [len(route) for route in routes]
        effective_limit = max(
            1,
            int(round(split_limit * (1.0 + (self.elite_overflow_ratio if overflow_ratio is None else overflow_ratio))))
        )
        fill_estimates = [
            min(1.0, box_count / effective_limit)
            for box_count in route_box_counts
        ]
        avg_fill_est = sum(fill_estimates) / len(fill_estimates) if fill_estimates else 0.0
        min_fill_est = min(fill_estimates) if fill_estimates else 0.0
        fill_std_est = (
            sum((fill - avg_fill_est) ** 2 for fill in fill_estimates) / len(fill_estimates)
        ) ** 0.5 if fill_estimates else 0.0
        tiny_route_count = sum(
            1
            for customer_count, fill_est in zip(route_customer_counts, fill_estimates)
            if customer_count <= self.tiny_route_customer_threshold
            or fill_est <= self.tiny_route_fill_threshold
        )
        overflow_route_count = sum(1 for box_count in route_box_counts if box_count > split_limit)
        return {
            "route_count": len(routes),
            "tiny_route_count": tiny_route_count,
            "avg_fill_rate": avg_fill_est,
            "min_fill_rate": min_fill_est,
            "route_fill_std": fill_std_est,
            "overflow_route_count": overflow_route_count,
        }

    def estimate_distance_limit(self, order, cust_box_map, depot, customers_map, split_limit):
        if not order:
            return None

        total_boxes = sum(cust_box_map.get(customer_id, 0) for customer_id in order)
        expected_route_count = max(1, round(total_boxes / max(1, split_limit)))
        giant_tour_distance = self.route_distance(depot, customers_map, order)
        return (giant_tour_distance / expected_route_count) * 1.15

    def generate_candidate_route_partitions(self, order, cust_box_map, depot, customers_map):
        order_key = tuple(order)
        if order_key in self.split_candidate_cache:
            return copy.deepcopy(self.split_candidate_cache[order_key])

        candidates = []
        seen_signatures = set()
        base_routes = self.decode_by_boxcount(
            order,
            cust_box_map,
            max_boxes_per_route=self.max_boxes_per_route,
        )
        base_route_count = len(base_routes)

        if self.dataset_scale() == "very_large":
            candidate_offsets = self.very_large_split_offsets
        elif self.dataset_scale() == "medium":
            candidate_offsets = [-5, 0, 1]
        else:
            candidate_offsets = self.split_offsets

        for offset in candidate_offsets:
            split_limit = max(1, self.max_boxes_per_route + int(offset))

            plain_routes = self.decode_by_boxcount(
                order,
                cust_box_map,
                max_boxes_per_route=split_limit,
            )
            plain_signature = tuple(tuple(route) for route in plain_routes)
            if plain_signature not in seen_signatures:
                if len(plain_routes) <= base_route_count + 1:
                    seen_signatures.add(plain_signature)
                    candidates.append(
                        {
                            "split_limit": split_limit,
                            "strategy": "box_threshold",
                            "distance_limit": None,
                            "routes": plain_routes,
                        }
                    )

            distance_limit = self.estimate_distance_limit(
                order,
                cust_box_map,
                depot,
                customers_map,
                split_limit,
            )
            adaptive_routes = self.decode_with_adaptive_splits(
                order,
                cust_box_map,
                depot,
                customers_map,
                max_boxes_per_route=max(1, int(round(split_limit * (1.0 + self.elite_overflow_ratio)))),
                distance_limit=distance_limit,
            )
            adaptive_signature = tuple(tuple(route) for route in adaptive_routes)
            if adaptive_signature not in seen_signatures:
                if len(adaptive_routes) <= base_route_count + 1:
                    seen_signatures.add(adaptive_signature)
                    candidates.append(
                        {
                            "split_limit": split_limit,
                            "strategy": "box_threshold_plus_distance",
                            "distance_limit": distance_limit,
                            "routes": adaptive_routes,
                        }
                    )

        if not candidates:
            candidates.append(
                {
                    "split_limit": self.max_boxes_per_route,
                    "strategy": "box_threshold",
                    "distance_limit": None,
                    "routes": base_routes,
                }
            )

        self.split_candidate_cache[order_key] = copy.deepcopy(candidates)
        return candidates

    def prescreen_candidate_partitions(self, candidates, cust_box_map):
        if len(candidates) <= 1:
            return candidates

        enriched = []
        for candidate in candidates:
            estimates = self.estimate_partition_metrics(
                candidate["routes"],
                cust_box_map,
                candidate["split_limit"],
            )
            enriched.append((candidate, estimates))

        best_route_count = min(item[1]["route_count"] for item in enriched)
        best_tiny_route_count = min(item[1]["tiny_route_count"] for item in enriched)
        best_min_fill = max(item[1]["min_fill_rate"] for item in enriched)
        best_fill_std = min(item[1]["route_fill_std"] for item in enriched)
        best_overflow = min(item[1]["overflow_route_count"] for item in enriched)

        filtered = []
        for candidate, estimates in enriched:
            route_count_slack = self.prescreen_route_count_slack()
            if estimates["route_count"] > best_route_count + route_count_slack:
                continue
            if estimates["tiny_route_count"] > best_tiny_route_count + 1:
                continue
            if self.dataset_scale() in {"large", "very_large"} and estimates["overflow_route_count"] > best_overflow + 1:
                continue
            if (
                estimates["route_fill_std"] > best_fill_std + 0.08
                and estimates["min_fill_rate"] < best_min_fill - 0.08
            ):
                continue
            candidate_copy = dict(candidate)
            candidate_copy["estimate"] = estimates
            filtered.append(candidate_copy)

        if not filtered:
            filtered = [dict(candidate, estimate=estimates) for candidate, estimates in enriched]

        if self.dataset_scale() in {"large", "very_large"}:
            filtered.sort(
                key=lambda candidate: (
                    candidate["estimate"]["overflow_route_count"],
                    candidate["estimate"]["tiny_route_count"],
                    candidate["estimate"]["route_fill_std"],
                    -candidate["estimate"]["min_fill_rate"],
                    candidate["estimate"]["route_count"],
                    -candidate["estimate"]["avg_fill_rate"],
                )
            )
        else:
            filtered.sort(
                key=lambda candidate: (
                    candidate["estimate"]["route_count"],
                    candidate["estimate"]["tiny_route_count"],
                    candidate["estimate"]["overflow_route_count"],
                    candidate["estimate"]["route_fill_std"],
                    -candidate["estimate"]["min_fill_rate"],
                    -candidate["estimate"]["avg_fill_rate"],
                )
            )

        return filtered[:max(1, self.max_full_candidate_evaluations)]

    def score_candidate(self, total_distance, infeasible_count, unpacked_boxes, route_count, fill_balance_penalty, route_balance_penalty, avg_fill_rate):
        policy = self.effective_policy()
        return (
            total_distance
            + (PENALTY_ALPHA * infeasible_count)
            + (unpacked_boxes * 100)
            + (route_count * policy["route_count_penalty"])
            + (fill_balance_penalty * policy["fill_balance_weight"])
            + (route_balance_penalty * 5)
            - (avg_fill_rate * policy["fill_reward_weight"])
        )

    def build_partition_summary(self, route_details, merged_route_count=0, overflow_route_count=0):
        route_fill_rates = [route["fill_rate"] for route in route_details]
        route_box_counts = [route["boxes_total"] for route in route_details]
        route_customer_counts = [len(route["route"]) for route in route_details]

        avg_fill_rate = sum(route_fill_rates) / len(route_fill_rates) if route_fill_rates else 0.0
        min_fill_rate = min(route_fill_rates) if route_fill_rates else 0.0
        max_fill_rate = max(route_fill_rates) if route_fill_rates else 0.0
        fill_balance_penalty = max_fill_rate - min_fill_rate
        route_fill_std = (
            sum((fill - avg_fill_rate) ** 2 for fill in route_fill_rates) / len(route_fill_rates)
        ) ** 0.5 if route_fill_rates else 0.0
        avg_route_boxes = sum(route_box_counts) / len(route_box_counts) if route_box_counts else 0.0
        min_route_boxes = min(route_box_counts) if route_box_counts else 0
        max_route_boxes = max(route_box_counts) if route_box_counts else 0
        route_balance_penalty = sum(
            abs(box_count - avg_route_boxes)
            for box_count in route_box_counts
        )
        avg_customers_per_route = sum(route_customer_counts) / len(route_customer_counts) if route_customer_counts else 0.0
        min_customers_per_route = min(route_customer_counts) if route_customer_counts else 0
        max_customers_per_route = max(route_customer_counts) if route_customer_counts else 0
        tiny_route_count = sum(
            1 for route in route_details
            if len(route["route"]) <= self.tiny_route_customer_threshold
            or route["fill_rate"] <= self.tiny_route_fill_threshold
        )

        return {
            "avg_fill_rate": avg_fill_rate,
            "min_fill_rate": min_fill_rate,
            "max_fill_rate": max_fill_rate,
            "fill_balance_penalty": fill_balance_penalty,
            "route_fill_std": route_fill_std,
            "avg_route_boxes": avg_route_boxes,
            "min_route_boxes": min_route_boxes,
            "max_route_boxes": max_route_boxes,
            "route_balance_penalty": route_balance_penalty,
            "avg_customers_per_route": avg_customers_per_route,
            "min_customers_per_route": min_customers_per_route,
            "max_customers_per_route": max_customers_per_route,
            "tiny_route_count": tiny_route_count,
            "merged_route_count": merged_route_count,
            "overflow_route_count": overflow_route_count,
            "avg_boxes_per_route": avg_route_boxes,
            "min_boxes_per_route": min_route_boxes,
            "max_boxes_per_route": max_route_boxes,
        }

    def score_partition_info(self, info):
        policy = self.effective_policy()
        return (
            info["total_distance"]
            + (PENALTY_ALPHA * info["infeasible_count"])
            + (info["unpacked_boxes"] * 100)
            + (info["route_count"] * policy["route_count_penalty"])
            + (info["tiny_route_count"] * policy["small_route_penalty"])
            + (info.get("overflow_route_count", 0) * policy["overflow_route_penalty"])
            + (info["route_fill_std"] * policy["balance_penalty_weight"])
            + (info["fill_balance_penalty"] * policy["fill_balance_weight"])
            + (info["route_balance_penalty"] * 5)
            - (info.get("merged_route_count", 0) * self.merged_route_reward)
            - (info["avg_fill_rate"] * policy["fill_reward_weight"])
        )

    def finalize_partition_info(self, info, routes, cust_box_map, split_limit, merged_route_count=0):
        finalized = dict(info)
        route_box_counts = self.route_box_counts(routes, cust_box_map)
        finalized["merged_route_count"] = int(merged_route_count)
        finalized["overflow_route_count"] = sum(1 for box_count in route_box_counts if box_count > split_limit)
        finalized["avg_boxes_per_route"] = finalized.get("avg_route_boxes", 0.0)
        finalized["min_boxes_per_route"] = finalized.get("min_route_boxes", 0)
        finalized["max_boxes_per_route"] = finalized.get("max_route_boxes", 0)
        return finalized, self.score_partition_info(finalized)

    def try_repair_routes(self, merged_path, routes, depot, customers_map, cust_box_map, split_limit):
        if not self.enable_tiny_route_repair:
            return [list(route) for route in routes if route], 0
        if len(routes) <= 1:
            return routes, 0

        repaired_routes = [list(route) for route in routes if route]
        merged_route_count = 0
        max_route_boxes = max(1, int(round(split_limit * (1.0 + self.elite_overflow_ratio))))
        changed = True

        while changed:
            changed = False
            current_score, current_info = self.evaluate_route_partition(
                merged_path,
                repaired_routes,
                depot,
                customers_map,
            )
            current_info, current_score = self.finalize_partition_info(
                current_info,
                repaired_routes,
                cust_box_map,
                split_limit,
                merged_route_count=merged_route_count,
            )
            route_details = []
            for route in repaired_routes:
                route_eval = self.route_eval_cache.get(tuple(route))
                if route_eval is None:
                    route_eval = evaluate_route(merged_path, route, use_packing=USE_PACKING)
                    self.route_eval_cache[tuple(route)] = route_eval
                route_details.append(
                    {
                        "route": route,
                        "fill_rate": route_eval.get("fill_rate", 0.0),
                        "boxes_total": route_eval.get("boxes_total", 0),
                    }
                )

            tiny_index = next(
                (
                    index for index, detail in enumerate(route_details)
                    if len(detail["route"]) <= self.tiny_route_customer_threshold
                    or detail["fill_rate"] <= self.tiny_route_fill_threshold
                ),
                None,
            )
            if tiny_index is None:
                break

            current_route = repaired_routes[tiny_index]
            neighbor_indices = [idx for idx in (tiny_index - 1, tiny_index + 1) if 0 <= idx < len(repaired_routes)]
            best_candidate = None

            for neighbor_index in neighbor_indices:
                base_neighbor = repaired_routes[neighbor_index]
                candidate_routes = [list(route) for route in repaired_routes]
                merged_route = base_neighbor + current_route
                if self.route_box_count(merged_route, cust_box_map) > max_route_boxes:
                    continue
                candidate_routes[neighbor_index] = merged_route
                del candidate_routes[tiny_index]
                candidate_score, candidate_info = self.evaluate_route_partition(
                    merged_path,
                    candidate_routes,
                    depot,
                    customers_map,
                )
                if candidate_info["infeasible_count"] > 0:
                    continue
                improvement = (
                    candidate_info["route_count"] < len(repaired_routes)
                    and candidate_info["min_fill_rate"] >= min(detail["fill_rate"] for detail in route_details)
                ) or candidate_info["total_distance"] <= sum(
                    self.route_distance(depot, customers_map, route) for route in repaired_routes
                ) * 1.03
                if not improvement:
                    continue
                score_improvement = candidate_score < current_score - 1e-6
                large_dataset_route_drop = (
                    self.dataset_scale() in {"large", "very_large"}
                    and candidate_info["route_count"] < current_info["route_count"]
                    and candidate_info["total_distance"] <= current_info["total_distance"] * self.route_drop_distance_ratio()
                )
                if not (score_improvement or large_dataset_route_drop):
                    continue
                if best_candidate is None or candidate_score < best_candidate[0]:
                    best_candidate = (candidate_score, candidate_routes)

            if best_candidate is not None:
                repaired_routes = best_candidate[1]
                merged_route_count += 1
                changed = True

        return repaired_routes, merged_route_count

    @staticmethod
    def candidate_route_customers(route):
        if not route:
            return []
        customers = [route[0], route[-1]]
        middle_customer = route[len(route) // 2]
        if middle_customer not in customers:
            customers.append(middle_customer)
        return customers

    def should_attempt_relocation(self, info):
        if self.dataset_scale() == "very_large":
            return (
                info.get("tiny_route_count", 0) > 0
                or info.get("route_fill_std", 0.0) > 0.035
                or info.get("min_fill_rate", 0.0) < max(0.0, info.get("avg_fill_rate", 0.0) - 0.04)
            )
        if self.dataset_scale() == "large":
            return (
                info.get("tiny_route_count", 0) > 0
                or info.get("route_fill_std", 0.0) > 0.025
                or info.get("min_fill_rate", 0.0) < max(0.0, info.get("avg_fill_rate", 0.0) - 0.03)
            )
        if self.dataset_scale() == "medium":
            return (
                info.get("tiny_route_count", 0) > 0
                or info.get("route_fill_std", 0.0) > 0.03
                or info.get("min_fill_rate", 0.0) < max(0.0, info.get("avg_fill_rate", 0.0) - 0.035)
            )
        return (
            info.get("tiny_route_count", 0) > 0
            or info.get("route_fill_std", 0.0) > 0.018
            or info.get("min_fill_rate", 0.0) < max(0.0, info.get("avg_fill_rate", 0.0) - 0.018)
        )

    def try_customer_relocation_repair(
        self,
        merged_path,
        routes,
        depot,
        customers_map,
        cust_box_map,
        split_limit,
        merged_route_count=0,
    ):
        if not self.enable_customer_relocation_repair or len(routes) < 2:
            return routes, 0

        best_routes = [list(route) for route in routes if route]
        best_score, best_info = self.evaluate_route_partition(
            merged_path,
            best_routes,
            depot,
            customers_map,
        )
        best_info, best_score = self.finalize_partition_info(
            best_info,
            best_routes,
            cust_box_map,
            split_limit,
            merged_route_count=merged_route_count,
        )
        if not self.should_attempt_relocation(best_info):
            return best_routes, 0
        additional_merged_routes = 0

        for _ in range(max(0, self.max_relocation_attempts)):
            route_details = list(best_info.get("routes") or [])
            if len(route_details) < 2:
                break

            weak_index = min(
                range(len(route_details)),
                key=lambda idx: (
                    route_details[idx].get("fill_rate", 0.0),
                    len(route_details[idx].get("route") or []),
                    route_details[idx].get("boxes_total", 0),
                ),
            )
            neighbor_indices = [
                idx for idx in (weak_index - 1, weak_index + 1)
                if 0 <= idx < len(best_routes)
            ]
            candidate_choice = None

            for neighbor_index in neighbor_indices:
                weak_route = best_routes[weak_index]
                neighbor_route = best_routes[neighbor_index]
                candidate_moves = []
                if len(weak_route) > 1:
                    for customer_id in self.candidate_route_customers(weak_route):
                        candidate_moves.append(("weak_to_neighbor", customer_id))
                for customer_id in self.candidate_route_customers(neighbor_route):
                    candidate_moves.append(("neighbor_to_weak", customer_id))

                for direction, customer_id in candidate_moves:
                    candidate_routes = [list(route) for route in best_routes]
                    weak_candidate = candidate_routes[weak_index]
                    neighbor_candidate = candidate_routes[neighbor_index]

                    if direction == "weak_to_neighbor":
                        if customer_id not in weak_candidate or len(weak_candidate) <= 1:
                            continue
                        weak_candidate.remove(customer_id)
                        if neighbor_index < weak_index:
                            neighbor_candidate.append(customer_id)
                        else:
                            neighbor_candidate.insert(0, customer_id)
                    else:
                        if customer_id not in neighbor_candidate:
                            continue
                        neighbor_candidate.remove(customer_id)
                        if weak_index < neighbor_index:
                            weak_candidate.append(customer_id)
                        else:
                            weak_candidate.insert(0, customer_id)

                    candidate_routes = [route for route in candidate_routes if route]
                    candidate_merged_routes = merged_route_count + max(0, len(best_routes) - len(candidate_routes))
                    candidate_score, candidate_info = self.evaluate_route_partition(
                        merged_path,
                        candidate_routes,
                        depot,
                        customers_map,
                    )
                    candidate_info, candidate_score = self.finalize_partition_info(
                        candidate_info,
                        candidate_routes,
                        cust_box_map,
                        split_limit,
                        merged_route_count=candidate_merged_routes,
                    )

                    if candidate_info["infeasible_count"] > best_info["infeasible_count"]:
                        continue
                    relocation_distance_increase_ratio = min(
                        self.max_distance_increase_ratio,
                        self.effective_policy()["relocation_distance_increase_ratio"],
                    )
                    distance_ratio_cap = relocation_distance_increase_ratio
                    if self.dataset_scale() == "very_large" and candidate_info["route_count"] >= best_info["route_count"]:
                        distance_ratio_cap = min(distance_ratio_cap, 1.0005)
                    elif self.dataset_scale() == "large" and candidate_info["route_count"] >= best_info["route_count"]:
                        distance_ratio_cap = min(distance_ratio_cap, 1.001)
                    if candidate_info["total_distance"] > best_info["total_distance"] * distance_ratio_cap:
                        continue
                    if candidate_info["overflow_route_count"] > best_info["overflow_route_count"] + 1:
                        continue

                    improved = (
                        candidate_info["route_count"] < best_info["route_count"]
                        or candidate_info["min_fill_rate"] > best_info["min_fill_rate"] + 1e-6
                        or candidate_info["route_fill_std"] < best_info["route_fill_std"] - 1e-6
                    )
                    if not improved:
                        continue
                    score_improvement = candidate_score < best_score - 1e-6
                    large_dataset_route_drop = (
                        self.dataset_scale() in {"large", "very_large"}
                        and candidate_info["route_count"] < best_info["route_count"]
                        and candidate_info["total_distance"] <= best_info["total_distance"] * self.route_drop_distance_ratio()
                    )
                    if not (score_improvement or large_dataset_route_drop):
                        continue

                    if candidate_choice is None or candidate_score < candidate_choice[0]:
                        candidate_choice = (
                            candidate_score,
                            candidate_routes,
                            candidate_info,
                            candidate_merged_routes,
                        )

            if candidate_choice is None:
                break

            best_score, best_routes, best_info, candidate_merged_routes = candidate_choice
            additional_merged_routes = max(0, candidate_merged_routes - merged_route_count)

        return best_routes, additional_merged_routes

    def evaluate_route_partition(self, merged_path, routes, depot, customers_map):
        total_distance = 0.0
        infeasible_count = 0
        feasible_routes = 0
        total_boxes = 0
        total_boxes_packed = 0
        packing_time = 0.0
        route_details = []

        for route in routes:
            route_distance = self.route_distance(depot, customers_map, route)
            total_distance += route_distance

            pack_start = time.perf_counter()
            route_key = tuple(route)
            if route_key in self.route_eval_cache:
                route_eval = self.route_eval_cache[route_key]
            else:
                route_eval = evaluate_route(
                    merged_path,
                    route,
                    use_packing=USE_PACKING
                )
                self.route_eval_cache[route_key] = route_eval
            packing_time += time.perf_counter() - pack_start

            feasible = route_eval.get("feasible", False)
            if feasible:
                feasible_routes += 1
            else:
                infeasible_count += 1

            total_boxes += route_eval.get("boxes_total", 0)
            total_boxes_packed += route_eval.get("boxes_packed", 0)
            route_details.append({
                "route": route,
                "distance": route_distance,
                "feasible": feasible,
                "boxes_total": route_eval.get("boxes_total", 0),
                "boxes_packed": route_eval.get("boxes_packed", 0),
                "fill_rate": route_eval.get("fill_rate", 0.0)
            })

        feasibility_rate = feasible_routes / len(routes) if routes else 0.0
        unpacked_boxes = max(0, total_boxes - total_boxes_packed)
        route_count = len(routes)
        summary = self.build_partition_summary(route_details)

        info = {
            "total_distance": total_distance,
            "infeasible_count": infeasible_count,
            "infeasible_routes": infeasible_count,
            "feasible_routes": feasible_routes,
            "feasibility_rate": feasibility_rate,
            "boxes_total": total_boxes,
            "boxes_packed": total_boxes_packed,
            "packing_time_seconds": packing_time,
            "routes": route_details,
            "route_count": route_count,
            "unpacked_boxes": unpacked_boxes,
        }
        info.update(summary)
        info["fill_rate"] = info["avg_fill_rate"]
        score = self.score_partition_info(info)
        return score, info

    def evaluate_permutation_fast(self, merged_path, perm, depot, customers_map, cust_box_map):
        routes = self.decode_by_boxcount(
            perm,
            cust_box_map,
            max_boxes_per_route=self.max_boxes_per_route
        )
        score, info = self.evaluate_route_partition(
            merged_path,
            routes,
            depot,
            customers_map,
        )
        info, score = self.finalize_partition_info(
            info,
            routes,
            cust_box_map,
            self.max_boxes_per_route,
            merged_route_count=0,
        )
        info["chosen_split_limit"] = self.max_boxes_per_route
        info["chosen_split_strategy"] = "fast_box_threshold"
        info["chosen_distance_limit"] = None
        info["number_of_split_candidates_tested"] = 1
        info["candidate_route_counts"] = [info["route_count"]]
        info["candidate_fill_rates"] = [info["avg_fill_rate"]]
        info["candidate_distances"] = [info["total_distance"]]
        info["candidate_split_summaries"] = [
            {
                "index": 1,
                "strategy": "fast_box_threshold",
                "split_limit": self.max_boxes_per_route,
                "distance_limit": None,
                "score": score,
                "route_count": info["route_count"],
                "avg_fill_rate": info["avg_fill_rate"],
                "min_fill_rate": info["min_fill_rate"],
                "distance": info["total_distance"],
                "infeasible_routes": info["infeasible_routes"],
                "unpacked_boxes": info["unpacked_boxes"],
            }
        ]
        return score, info

    def evaluate_final_best_refinement(self, merged_path, perm, depot, customers_map, cust_box_map):
        if not self.enable_final_best_refinement:
            return self.evaluate_permutation_fast(merged_path, perm, depot, customers_map, cust_box_map)
        final_split_offset = self.final_refinement_split_offset
        final_overflow_ratio = self.final_refinement_overflow_ratio
        if self.is_very_large_dataset():
            final_split_offset = min(final_split_offset, 4)
            final_overflow_ratio = min(final_overflow_ratio, 0.10)
        elif self.dataset_scale() == "medium":
            final_split_offset = min(final_split_offset, 3)
            final_overflow_ratio = min(final_overflow_ratio, 0.08)
        split_limit = max(1, self.max_boxes_per_route + final_split_offset)
        distance_limit = self.estimate_distance_limit(
            perm,
            cust_box_map,
            depot,
            customers_map,
            split_limit,
        )
        aggressive_routes = self.decode_with_adaptive_splits(
            perm,
            cust_box_map,
            depot,
            customers_map,
            max_boxes_per_route=max(1, int(round(split_limit * (1.0 + final_overflow_ratio)))),
            distance_limit=distance_limit,
        )
        aggressive_routes, merged_route_count = self.try_repair_routes(
            merged_path,
            aggressive_routes,
            depot,
            customers_map,
            cust_box_map,
            split_limit,
        )
        aggressive_routes, relocation_merged_count = self.try_customer_relocation_repair(
            merged_path,
            aggressive_routes,
            depot,
            customers_map,
            cust_box_map,
            split_limit,
            merged_route_count=merged_route_count,
        )
        merged_route_count += relocation_merged_count
        score, info = self.evaluate_route_partition(
            merged_path,
            aggressive_routes,
            depot,
            customers_map,
        )
        info, score = self.finalize_partition_info(
            info,
            aggressive_routes,
            cust_box_map,
            split_limit,
            merged_route_count=merged_route_count,
        )
        info["chosen_split_limit"] = split_limit
        info["chosen_split_strategy"] = "final_extra_refinement"
        info["chosen_distance_limit"] = distance_limit
        info["number_of_split_candidates_tested"] = 1
        info["candidate_route_counts"] = [info["route_count"]]
        info["candidate_fill_rates"] = [info["avg_fill_rate"]]
        info["candidate_distances"] = [info["total_distance"]]
        info["candidate_split_summaries"] = [
            {
                "index": 1,
                "strategy": "final_extra_refinement",
                "split_limit": split_limit,
                "distance_limit": distance_limit,
                "score": score,
                "route_count": info["route_count"],
                "avg_fill_rate": info["avg_fill_rate"],
                "min_fill_rate": info["min_fill_rate"],
                "distance": info["total_distance"],
                "infeasible_routes": info["infeasible_routes"],
                "unpacked_boxes": info["unpacked_boxes"],
            }
        ]
        return score, info

    def evaluate_permutation(self, merged_path, perm, adaptive=True, final_refinement=False):
        inst_name, container, customers, boxes = load_merged(merged_path)
        depot_customer, real_customers = split_depot_and_customers(customers)
        if not self.num_customers:
            self.num_customers = len(real_customers)

        cust_box_map = self.build_customer_boxcount_map(real_customers)
        depot = (depot_customer["x"], depot_customer["y"]) if depot_customer else (0, 0)
        customers_map = {c["customer_id"]: c for c in real_customers}
        perm_key = tuple(perm)
        cache_mode = "adaptive_final" if adaptive and final_refinement else ("adaptive" if adaptive else "fast")
        cache_key = (cache_mode, perm_key)

        if cache_key in self.permutation_eval_cache:
            cached_score, cached_info = self.permutation_eval_cache[cache_key]
            return cached_score, copy.deepcopy(cached_info)

        if not adaptive or not self.enable_adaptive_decoding:
            fast_score, fast_info = self.evaluate_permutation_fast(
                merged_path,
                perm,
                depot,
                customers_map,
                cust_box_map,
            )
            self.permutation_eval_cache[cache_key] = (fast_score, copy.deepcopy(fast_info))
            return fast_score, fast_info

        candidate_partitions = self.generate_candidate_route_partitions(
            perm,
            cust_box_map,
            depot,
            customers_map,
        )
        if final_refinement and self.enable_final_best_refinement:
            final_split_offset = self.final_refinement_split_offset
            final_overflow_ratio = self.final_refinement_overflow_ratio
            if self.is_very_large_dataset():
                final_split_offset = min(final_split_offset, 4)
                final_overflow_ratio = min(final_overflow_ratio, 0.10)
            elif self.dataset_scale() == "medium":
                final_split_offset = min(final_split_offset, 3)
                final_overflow_ratio = min(final_overflow_ratio, 0.08)
            split_limit = max(1, self.max_boxes_per_route + final_split_offset)
            distance_limit = self.estimate_distance_limit(
                perm,
                cust_box_map,
                depot,
                customers_map,
                split_limit,
            )
            candidate_partitions.append(
                {
                    "split_limit": split_limit,
                    "strategy": "final_extra_refinement",
                    "distance_limit": distance_limit,
                    "routes": self.decode_with_adaptive_splits(
                        perm,
                        cust_box_map,
                        depot,
                        customers_map,
                        max_boxes_per_route=max(
                            1,
                            int(round(split_limit * (1.0 + final_overflow_ratio))),
                        ),
                        distance_limit=distance_limit,
                    ),
                }
            )
        candidate_partitions = self.prescreen_candidate_partitions(candidate_partitions, cust_box_map)

        best_score = float("inf")
        best_info = None
        candidate_summaries = []

        if self.verbose:
            print(
                f"[{self.progress_label}] Prescreen kept {len(candidate_partitions)} adaptive candidates",
            )

        for index, candidate in enumerate(candidate_partitions, start=1):
            repaired_routes, merged_route_count = self.try_repair_routes(
                merged_path,
                candidate["routes"],
                depot,
                customers_map,
                cust_box_map,
                candidate["split_limit"],
            )
            relocation_merged_count = 0
            if self.enable_customer_relocation_repair and self.should_attempt_relocation(candidate.get("estimate", {})):
                repaired_routes, relocation_merged_count = self.try_customer_relocation_repair(
                    merged_path,
                    repaired_routes,
                    depot,
                    customers_map,
                    cust_box_map,
                    candidate["split_limit"],
                    merged_route_count=merged_route_count,
                )
                merged_route_count += relocation_merged_count
            candidate_score, candidate_info = self.evaluate_route_partition(
                merged_path,
                repaired_routes,
                depot,
                customers_map,
            )
            candidate_info, candidate_score = self.finalize_partition_info(
                candidate_info,
                repaired_routes,
                cust_box_map,
                candidate["split_limit"],
                merged_route_count=merged_route_count,
            )
            candidate_summary = {
                "index": index,
                "strategy": candidate["strategy"],
                "split_limit": candidate["split_limit"],
                "distance_limit": candidate["distance_limit"],
                "score": candidate_score,
                "route_count": candidate_info["route_count"],
                "avg_fill_rate": candidate_info["avg_fill_rate"],
                "min_fill_rate": candidate_info["min_fill_rate"],
                "distance": candidate_info["total_distance"],
                "infeasible_routes": candidate_info["infeasible_routes"],
                "unpacked_boxes": candidate_info["unpacked_boxes"],
            }
            candidate_summaries.append(candidate_summary)

            if self.verbose:
                distance_limit_text = (
                    "none"
                    if candidate["distance_limit"] is None
                    else f"{candidate['distance_limit']:.2f}"
                )
                print(
                    f"[{self.progress_label}] Split candidate {index}/{len(candidate_partitions)} "
                    f"strategy={candidate['strategy']} split_limit={candidate['split_limit']} "
                    f"distance_limit={distance_limit_text} score={candidate_score:.2f} "
                    f"routes={candidate_info['route_count']} avg_fill={candidate_info['avg_fill_rate']:.4f} "
                    f"min_fill={candidate_info['min_fill_rate']:.4f} distance={candidate_info['total_distance']:.2f}"
                )

            if candidate_score < best_score:
                best_score = candidate_score
                best_info = dict(candidate_info)
                best_info["chosen_split_limit"] = candidate["split_limit"]
                best_info["chosen_split_strategy"] = candidate["strategy"]
                best_info["chosen_distance_limit"] = candidate["distance_limit"]

        if best_info is None:
            best_info = {
                "total_distance": 0.0,
                "infeasible_count": 0,
                "infeasible_routes": 0,
                "feasible_routes": 0,
                "feasibility_rate": 0.0,
                "boxes_total": 0,
                "boxes_packed": 0,
                "fill_rate": 0.0,
                "avg_fill_rate": 0.0,
                "min_fill_rate": 0.0,
                "max_fill_rate": 0.0,
                "packing_time_seconds": 0.0,
                "routes": [],
                "route_count": 0,
                "unpacked_boxes": 0,
                "fill_balance_penalty": 0.0,
                "route_balance_penalty": 0.0,
                "route_fill_std": 0.0,
                "avg_route_boxes": 0.0,
                "min_route_boxes": 0,
                "max_route_boxes": 0,
                "avg_customers_per_route": 0.0,
                "min_customers_per_route": 0,
                "max_customers_per_route": 0,
                "tiny_route_count": 0,
                "merged_route_count": 0,
                "overflow_route_count": 0,
                "avg_boxes_per_route": 0.0,
                "min_boxes_per_route": 0,
                "max_boxes_per_route": 0,
                "chosen_split_limit": self.max_boxes_per_route,
                "chosen_split_strategy": "none",
                "chosen_distance_limit": None,
            }

        best_info["number_of_split_candidates_tested"] = len(candidate_summaries)
        best_info["candidate_route_counts"] = [summary["route_count"] for summary in candidate_summaries]
        best_info["candidate_fill_rates"] = [summary["avg_fill_rate"] for summary in candidate_summaries]
        best_info["candidate_distances"] = [summary["distance"] for summary in candidate_summaries]
        best_info["candidate_split_summaries"] = candidate_summaries

        if self.verbose and candidate_summaries:
            winner = min(candidate_summaries, key=lambda summary: summary["score"])
            print(
                f"[{self.progress_label}] Winning split strategy={winner['strategy']} "
                f"split_limit={winner['split_limit']} score={winner['score']:.2f} "
                f"routes={winner['route_count']} avg_fill={winner['avg_fill_rate']:.4f}"
            )

        self.permutation_eval_cache[cache_key] = (best_score, copy.deepcopy(best_info))
        return best_score, best_info

    def run(self, merged_path):
        random.seed(self.seed)
        self.route_eval_cache = {}
        self.permutation_eval_cache = {}
        self.split_candidate_cache = {}

        overall_start = time.perf_counter()

        inst_name, container, customers, boxes = load_merged(merged_path)
        depot_customer, real_customers = split_depot_and_customers(customers)
        customer_ids = [c["customer_id"] for c in real_customers]
        self.num_customers = len(real_customers)
        cust_box_map = self.build_customer_boxcount_map(real_customers)
        n = len(customer_ids)

        population = [random.sample(customer_ids, n) for _ in range(self.pop_size - 2)]
        population.append(customer_ids[:])
        population.append(list(reversed(customer_ids)))

        best_solution = None
        best_score = float("inf")
        best_info = None
        history = []
        generation_start = overall_start

        for generation in range(self.gens):
            preliminary_population = []

            for individual in population:
                score, info = self.evaluate_permutation(merged_path, individual, adaptive=False)
                preliminary_population.append((score, individual, info))

            preliminary_population.sort(key=lambda x: x[0])
            scored_population = preliminary_population[:]

            should_run_adaptive = self.enable_adaptive_decoding and (
                self.adaptive_every_generations <= 1
                or generation % self.adaptive_every_generations == 0
                or generation == self.gens - 1
            )
            adaptive_count = min(
                len(scored_population),
                max(self.adaptive_top_min, int(round(self.pop_size * self.adaptive_top_fraction))),
            )

            if should_run_adaptive and adaptive_count > 0:
                for index in range(adaptive_count):
                    _, individual, _ = scored_population[index]
                    adaptive_score, adaptive_info = self.evaluate_permutation(
                        merged_path,
                        individual,
                        adaptive=True,
                    )
                    scored_population[index] = (adaptive_score, individual, adaptive_info)

            scored_population.sort(key=lambda x: x[0])

            for score, individual, info in scored_population:
                if score < best_score:
                    best_score = score
                    best_solution = individual[:]
                    best_info = info

            if self.verbose and should_run_adaptive:
                print(
                    f"[{self.progress_label}] Adaptive decoding refined top {adaptive_count} "
                    f"individuals in generation {generation + 1}"
                )

            elite_count = max(4, int(0.05 * self.pop_size))
            new_population = [scored_population[i][1][:] for i in range(elite_count)]

            while len(new_population) < self.pop_size:
                scores_only = [x[0] for x in scored_population]
                population_only = [x[1] for x in scored_population]

                parent1 = tournament_select(population_only, scores_only)
                parent2 = tournament_select(population_only, scores_only)

                if random.random() < self.cx_prob:
                    child = order_crossover(parent1, parent2)
                else:
                    child = parent1[:]

                if USE_ENHANCED_MUTATION:
                    child = hybrid_mutation(
                        child,
                        self.mut_prob,
                        decoder=self.decode_by_boxcount,
                        cust_boxcount_map=cust_box_map,
                        max_boxes_per_route=self.max_boxes_per_route,
                        enable_route_balance_mutation=self.enable_route_balance_mutation,
                        route_balance_probability=self.effective_policy()["route_balance_mutation_probability"],
                    )
                else:
                    child = swap_mutation(child, self.mut_prob)
               
                new_population.append(child)

            population = new_population
            history.append(best_score)

            if self.verbose and (generation % 10 == 0 or generation == self.gens - 1):
                elapsed = time.perf_counter() - generation_start
                print(
                    f"[{self.progress_label}] Generation {generation + 1}/{self.gens} "
                    f"Best Score = {best_score:.2f} Elapsed = {elapsed:.2f}s"
                )

        if best_solution is not None:
            best_score, best_info = self.evaluate_permutation(merged_path, best_solution, adaptive=self.enable_adaptive_decoding)
            if self.enable_final_best_refinement:
                inst_name, container, customers, boxes = load_merged(merged_path)
                depot_customer, real_customers = split_depot_and_customers(customers)
                cust_box_map = self.build_customer_boxcount_map(real_customers)
                depot = (depot_customer["x"], depot_customer["y"]) if depot_customer else (0, 0)
                customers_map = {c["customer_id"]: c for c in real_customers}
                final_score, final_info = self.evaluate_final_best_refinement(
                    merged_path,
                    best_solution,
                    depot,
                    customers_map,
                    cust_box_map,
                )
                final_policy = self.effective_policy()
                route_drop_distance_ratio = self.route_drop_distance_ratio()
                should_take_final_refinement = (
                    final_score < best_score
                    and (
                        (
                            final_info["route_count"] < best_info["route_count"]
                            and final_info["total_distance"]
                            <= best_info["total_distance"] * route_drop_distance_ratio
                        )
                        or (
                            final_info["min_fill_rate"] > best_info["min_fill_rate"] + 1e-6
                            and final_info["total_distance"]
                            <= best_info["total_distance"] * final_policy["final_refinement_distance_ratio"]
                        )
                        or final_info["total_distance"]
                        <= best_info["total_distance"] * min(
                            1.0,
                            final_policy["final_refinement_distance_ratio"] - 0.002,
                        )
                    )
                )
                if should_take_final_refinement:
                    best_score, best_info = final_score, final_info

        runtime_seconds = time.perf_counter() - overall_start

        result = {
            "model": MODEL_NAME,
            "best_order": best_solution,
            "best_score": best_score,
            "best_info": best_info,
            "history": history,
            "runtime_seconds": runtime_seconds,
            "duration": runtime_seconds,
        }

        return result

