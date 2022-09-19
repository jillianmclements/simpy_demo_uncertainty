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
import json
import simpy
from scipy import signal 


# simulation time step (days)
TIME_STEP = 1


class HospitalSim:
    def __init__(
        self,
        initial_patients: int,
        initial_beds: int,
        random_seed: int,
        # simulate a disturbance in the arrival rate (e.g.,)
        disturbance_amplitude: int = 100,
        disturbance_length: int = 60,
        disturbance_start: int = 100,
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
        self.disturbance_length = disturbance_length
        self.disturbance_amplitude = disturbance_amplitude
        self.disturbance_start = disturbance_start

        self.reset(
            initial_beds=initial_beds, 
            initial_patients=initial_patients, 
            random_seed=random_seed,
            disturbance_length=disturbance_length,
            disturbance_amplitude=disturbance_amplitude,
            disturbance_start=disturbance_start
            )

    def reset(
        self, 
        initial_patients: int, 
        initial_beds: int,
        random_seed: int,
        disturbance_length: int,
        disturbance_amplitude: int,
        disturbance_start: int,
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

        # create disturbance signal
        self.mean_arrival_signal = self._create_disturbance_signal(
            disturbance_length=disturbance_length,
            disturbance_amplitude=disturbance_amplitude,
            disturbance_start=disturbance_start,
        )

    def _create_disturbance_signal(
        self,
        disturbance_length: int,
        disturbance_amplitude: int,
        disturbance_start: int,
        ):
        # create a signal with a length of `disturbance_length` days
        # and an amplitude of `disturbance_amplitude` mean patients per day

        # normal week signal
        weekly_signal = np.ones(365) * self.mean_arrivals_weekday
        weekly_signal[::6] = self.mean_arrivals_weekend
        weekly_signal[::7] = self.mean_arrivals_weekend

        # add disturbance to the signal
        disturbance_signal = disturbance_amplitude*signal.gaussian(disturbance_length, std=1)
        pad_before = disturbance_start
        pad_after = 365 - (disturbance_length + disturbance_start)
        disturbance_signal = np.pad(disturbance_signal, (pad_before, pad_after))
        
        # combine the signals
        mean_arrival_signal = weekly_signal + disturbance_signal

        return mean_arrival_signal

    def _adjust_bed_numbers(self, Δnum_beds: int):
        # adjust the number of beds by Δnum_beds

        yield self.env.timeout(self.delay_to_change_beds) # wait for delay

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

    # def _get_day_of_week(self) -> int:
    #     return int((self.env.now) % 7)

    # def _is_weekend(self) -> bool:
    #     return self._get_day_of_week() >= 5

    def _get_time_to_next_arrival(self) -> float:
        # time until next patient admission

        # if self._is_weekend():
        #     mean_arrivals_per_day = self.mean_arrivals_weekend
        # else:
        #     mean_arrivals_per_day = self.mean_arrivals_weekday

        mean_arrivals_per_day = self.mean_arrival_signal[int(self.env.now)]

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
            "disturbance_length": self.disturbance_length,
            "disturbance_amplitude": self.disturbance_amplitude,
            "disturbance_start": self.disturbance_start,
        }


if __name__ == "__main__":
    
    # DataFrame to store state, action, config
    data = pd.DataFrame() 

    # number of episodes for benchmark test set
    configs = json.load(open('simpy_demo/assessment_40episodes.json'))
    configs = configs["episodeConfigurations"]

    # choose episode length (in days)
    episode_length = 365

    for config in configs:
        # initialize sim
        sim = HospitalSim(**config)

        # run sim for an episode and log results for benchmark controller
        for iter in range(episode_length):
            # get current state
            state = sim.get_current_state()

            # simple controller (action)
            if state['utilization'] >= 0.9:
                Δnum_beds = +25
            elif state['utilization'] < 0.7:
                Δnum_beds = -10
            else:
                Δnum_beds = 0

            # NOTE: it takes 1 day to change the number of beds
            sim.step(Δnum_beds=Δnum_beds)

            # Log results
            config = sim.get_current_config()
            state.update(config)
            state.update({'change_beds': Δnum_beds})
            data = pd.concat([data, pd.DataFrame([state])], ignore_index=True)

    # save data for comparison to DRL agent ("brain")
    data.to_csv('benchmark_results.csv', index=False)