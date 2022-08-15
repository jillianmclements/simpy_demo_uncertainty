"""
Base simulation to model hospital bed utilization.

Notes
-----
All durations are in units of days.
"""


import random
from typing import Dict
import numpy as np
import pandas as pd

import simpy


# simulation time step (days)
TIME_STEP = 1


class HospitalSim:
    def __init__(
        self,
        initial_patients: int,
        initial_beds: int,
        random_seed: int,
        mean_arrivals_weekday: float = 90,
        mean_arrivals_weekend: float = 120,
        mean_length_of_stay: float = 3,
        # can change up to `max_bed_change` beds per time step
        max_bed_change: int = 25,
        # how long it takes to add or remove beds
        delay_to_change_beds: int = 1,
    ):

        self.mean_arrivals_weekday = mean_arrivals_weekday
        self.mean_arrivals_weekend = mean_arrivals_weekend
        self.mean_length_of_stay = mean_length_of_stay

        self.max_bed_change = max_bed_change
        self.delay_to_change_beds = delay_to_change_beds

        # scenario configuration
        self.initial_patients = initial_patients
        self.initial_beds = initial_beds
        self.random_seed = random_seed

        self.reset(initial_beds=initial_beds, initial_patients=initial_patients, random_seed=random_seed)

    def reset(
        self, 
        initial_patients: int, 
        initial_beds: int,
        random_seed: int
    ):
        # set seed to control uncertainty
        random.seed(random_seed)

        # initialize environment
        self.env = simpy.Environment()

        # counter for reporting daily simulation state
        self.next_time_step = 0

        # number of beds at beginning of episodes
        self.num_beds = initial_beds

        # number of patients currently receiving care
        self.num_patients = initial_patients

        # number of patients not admitted because there were no beds available
        self.num_patients_overflow = 0

        # load initial batch of patients
        for _ in range(self.num_patients):
            self.env.process(self._process_patient(inital_batch=True))

        # start process for new patient arrivals
        self.env.process(self._patient_arrivals())

    def _adjust_bed_numbers(self, Δnum_beds: int):

        yield self.env.timeout(self.delay_to_change_beds)

        self.num_beds += Δnum_beds

    def _get_length_of_stay(self, initial_batch: bool = False) -> float:

        patient_length_of_stay = random.expovariate(1 / self.mean_length_of_stay)

        if initial_batch:
            # remaining length of stay (since the initial batch of patients
            # were admitted prior to the start of the simulation)
            patient_length_of_stay *= random.random()

        return patient_length_of_stay

    def _process_patient(self, inital_batch: bool = False):

        if self.num_beds > self.num_patients:

            # the patient is admitted to the hospital and occupies a bed
            self.num_patients += 1

            yield self.env.timeout(self._get_length_of_stay(initial_batch=inital_batch))

            # the patient is discharged from the hospital and frees up a bed
            self.num_patients -= 1

        else:

            # no beds are available
            self.num_patients_overflow += 1

    def _get_day_of_week(self) -> int:
        return int((self.env.now) % 7)

    def _is_weekend(self) -> bool:
        return self._get_day_of_week() >= 5

    def _get_time_to_next_arrival(self) -> float:
        # time until next patient admission

        if self._is_weekend():
            mean_arrivals_per_day = self.mean_arrivals_weekend
        else:
            mean_arrivals_per_day = self.mean_arrivals_weekday

        inter_arrival_time = 1 / mean_arrivals_per_day
        return random.expovariate(1 / inter_arrival_time)

    def _patient_arrivals(self):

        while True:

            self.env.process(self._process_patient())

            yield self.env.timeout(self._get_time_to_next_arrival())

    def step(self, Δnum_beds: int = 0):

        assert (
            Δnum_beds <= self.max_bed_change
        ), "Can only change up to 25 beds per day."

        self.env.process(self._adjust_bed_numbers(Δnum_beds))

        # reset patient overflow counter for the day
        self.num_patients_overflow = 0

        self.next_time_step += TIME_STEP
        self.env.run(until=self.next_time_step)

    def get_current_state(self) -> Dict:
        return {
            "simulation_time": self.env.now,
            "num_beds": self.num_beds,
            "num_patients": self.num_patients,
            "num_patients_overflow": self.num_patients_overflow,
            "utilization": self.num_patients / self.num_beds,
        }

    def get_current_config(self) -> Dict:
        return {
            "initial_patients": self.initial_patients,
            "initial_beds": self.initial_beds,
            "random_seed": self.random_seed,
        }


if __name__ == "__main__":
    
    # DataFrame to store state, action, config
    data = pd.DataFrame() 

    # choose 10 random seeds for benchmark
    seeds = np.random.randint(0,100,size=10)

    # choose episode length (in days)
    episode_length = 2*365

    for seed in seeds:
        DEFAULT_CONFIG = {"initial_beds": 200, "initial_patients": 0, "random_seed": seed.item()}
        sim = HospitalSim(**DEFAULT_CONFIG)

        # run sim for an episode and log results for benchmark controller
        for iter in range(episode_length):
            # get current state
            state = sim.get_current_state()
            config = sim.get_current_config()
            state.update(config)
            data = pd.concat([data, pd.DataFrame([state])], ignore_index=True)

            # simple controller (action)
            if state['utilization'] >= 0.9:
                Δnum_beds = +25
            elif state['utilization'] < 0.7:
                Δnum_beds = -10
            else:
                Δnum_beds = 0

            # NOTE: it takes 1 day to change the number of beds
            sim.step(Δnum_beds=Δnum_beds)

    # save data for comparison to DRL agent ("brain")
    data.to_csv('benchmark_results.csv', index=False)