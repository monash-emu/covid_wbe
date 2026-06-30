import pandas as pd
import numpy as np
from jax import Array, numpy as jnp

from typing import NamedTuple

from summer3.epi import CompartmentMap, Stratification, InfectionProcess, CompartmentalEpiModel
from summer3.graph import Parameter, Time, CompartmentValues, defer
from summer3.computegraph import Function
from summer3.proto import TransitionFlow, EntryFlow
from summer3.categories import Category, CategoryGroup, CategoryData

class ModelSpec(NamedTuple):
    random_process_step: int
    times: pd.DatetimeIndex
    population: float

class RandomProcessSpec(NamedTuple):
    default_values: np.ndarray
    rp_func: Function

class ModelInfo(NamedTuple):
    model: CompartmentalEpiModel
    random_process_spec: RandomProcessSpec

def get_random_process(times: pd.DatetimeIndex, step: int) -> Function:
    """
    Return a Computegraph Function that returns the exponential of the interpolated value of the random process at the given time.
    Args:
        times: The times at which the random process is evaluated.
        step: The step size of the random process.
    Returns:
        A Function that returns the exponential of the interpolated value of the random process at the given time.
    """
    proc_x = np.arange(0, len(times) + step, step)
    proc_vals_ref = np.zeros(len(proc_x))

    cum_proc_vals = defer(jnp.cumsum)(Parameter("proc_vals", proc_vals_ref))

    def random_process(t: float, process_values: Array) -> Array:
        """
        Return the exponential of the interpolated value of process_values at t.
        Args:
            t: Time (x axis)
            process_values: The (cumulative) values of the random process.
        Returns:
            float
        """
        return jnp.exp(
            jnp.interp(
                t, proc_x, process_values, left=process_values[0], right=process_values[-1]
            )
        )

    rp_func = defer(random_process)(Time, cum_proc_vals).set_name("random_process")
    return RandomProcessSpec(proc_vals_ref, rp_func)

def seed_func(t: float, start_time: float, duration: float, seed_rate: float) -> float:
    t_offset = t - start_time
    return jnp.where(t_offset < 0.0, 0.0, jnp.where(t_offset < duration, seed_rate, 0.0))

def get_seed_func() -> Function:
    return defer(seed_func)(Time, Parameter("seed_start_time", 0.0), Parameter("seed_duration", 7.0), Parameter("seed_rate", 1.0)).set_name("seed")

def get_model(spec: ModelSpec) -> CompartmentalEpiModel:
    disease_state = Stratification("disease_state", ["S", "E", "I", "R"])
    humans = CompartmentMap.new(disease_state)

    all_comps = Category(disease_state[...])
    cgroup = CategoryGroup([all_comps])

    rp_spec = get_random_process(spec.times, spec.random_process_step)

    eff_contact_rate = Parameter("contact_rate", 0.2) * rp_spec.rp_func

    iproc = defer(InfectionProcess)(cgroup, cgroup, disease_state["I"])
    foi = defer(InfectionProcess.process)(iproc, CompartmentValues, eff_contact_rate)

    infection = TransitionFlow("infection", disease_state["S"], disease_state["E"], foi)
    progression = TransitionFlow("progression", disease_state["E"], disease_state["I"], 1.0 / Parameter("latent_time", 3.0))
    recovery = TransitionFlow("recovery", disease_state["I"], disease_state["R"], 1.0 / Parameter("recovery_time", 6.5))
    waning = TransitionFlow("waning", disease_state["R"], disease_state["S"], 1.0 / Parameter("waning_time", 100.0))
    
    seed = EntryFlow("seed", disease_state["E"], get_seed_func())

    epi_model = CompartmentalEpiModel(humans, spec.times)
    epi_model.add_flow(seed)
    epi_model.add_flow(infection)
    epi_model.add_flow(progression)
    epi_model.add_flow(recovery)
    epi_model.add_flow(waning)

    base_pops = CategoryData(disease_state.categories(), jnp.array([spec.population, 0.0, 0.0, 0.0]))
    epi_model.set_initial_population(base_pops)

    return ModelInfo(epi_model, rp_spec)