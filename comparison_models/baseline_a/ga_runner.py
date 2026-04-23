# comparison_models/baseline_a/ga_runner.py

import time
import random
from math import sqrt

from comparison_models.common.algorithms.crossover import order_crossover
from comparison_models.baseline_a.mutation import swap_mutation
from comparison_models.common.algorithms.selection import tournament_select
from comparison_models.common.loaders.route_evaluator import load_merged, split_depot_and_customers
from comparison_models.baseline_a.config import PENALTY_ALPHA, MODEL_NAME


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

    def evaluate_permutation(self, merged_path, perm):
        inst_name, container, customers, boxes = load_merged(merged_path)
        depot_customer, real_customers = split_depot_and_customers(customers)

        cust_box_map = self.build_customer_boxcount_map(real_customers)
        routes = self.decode_by_boxcount(
            perm,
            cust_box_map,
            max_boxes_per_route=self.max_boxes_per_route
        )

        depot = (depot_customer["x"], depot_customer["y"]) if depot_customer else (0, 0)
        customers_map = {c["customer_id"]: c for c in real_customers}

        total_distance = 0.0
        infeasible_count = 0
        feasible_routes = 0
        total_boxes = 0
        total_boxes_packed = 0
        total_fill_rate = 0.0
        packing_time = 0.0

        route_details = []

        for route in routes:
            route_distance = self.route_distance(depot, customers_map, route)
            total_distance += route_distance

            route_boxes_total = sum(
                len(customers_map[customer_id].get("assigned_boxes", []))
                for customer_id in route
            )
            total_boxes += route_boxes_total

            route_details.append({
                "route": route,
                "distance": route_distance,
                "feasible": None,
                "boxes_total": route_boxes_total,
                "boxes_packed": 0,
                "fill_rate": 0.0
            })

        avg_fill_rate = total_fill_rate / len(routes) if routes else 0.0
        feasibility_rate = feasible_routes / len(routes) if routes else 0.0

        score = total_distance if infeasible_count == 0 else PENALTY_ALPHA * infeasible_count + total_distance

        return score, {
            "total_distance": total_distance,
            "infeasible_count": infeasible_count,
            "infeasible_routes": infeasible_count,
            "feasible_routes": feasible_routes,
            "feasibility_rate": feasibility_rate,
            "boxes_total": total_boxes,
            "boxes_packed": total_boxes_packed,
            "fill_rate": avg_fill_rate,
            "packing_time_seconds": packing_time,
            "routes": route_details,
        }

    def run(self, merged_path):
        random.seed(self.seed)

        overall_start = time.perf_counter()

        inst_name, container, customers, boxes = load_merged(merged_path)
        depot_customer, real_customers = split_depot_and_customers(customers)
        customer_ids = [c["customer_id"] for c in real_customers]
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
            scored_population = []

            for individual in population:
                score, info = self.evaluate_permutation(merged_path, individual)
                scored_population.append((score, individual, info))

                if score < best_score:
                    best_score = score
                    best_solution = individual[:]
                    best_info = info

            scored_population.sort(key=lambda x: x[0])

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

