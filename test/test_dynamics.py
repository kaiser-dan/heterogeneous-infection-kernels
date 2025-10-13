import sys

import numpy as np

import pytest

import src.dynamics

from src._logger import logger
logger.add(sys.stderr, level="DEBUG")


class TestActivations:
    def test_no_contagion(self):
        # (Psuedo-)random global parameters
        N = 37
        T = 100

        # Random network topology
        adjacency = np.random.randint(low=0, high=2, size=(N, N))  # high is exclusive
        kmax = max(np.sum(adjacency, axis=0))

        # No contagion
        states = np.zeros((N, T))

        # Expected output
        expected_infections = np.zeros((N, kmax))
        expected_noninfections = np.zeros((N, kmax))

        # Compute activations
        actual_infections, actual_noninfections = src.dynamics.get_activations(states, adjacency, kmax)

        logger.debug(f"Infection events:\n {actual_infections}")
        logger.debug(f"Non-infection events:\n {actual_noninfections}")

        assert np.equal(actual_infections, expected_infections).all()
        assert np.equal(actual_noninfections, expected_noninfections).all()

    def test_trivial_contagion(self):
        # (Psuedo-)random global parameters
        N = 37
        T = 100

        # Random network topology
        adjacency = np.random.randint(low=0, high=2, size=(N, N))  # high is exclusive
        kmax = max(np.sum(adjacency, axis=0))

        # Trivial contagion
        states = np.ones((N, T))

        # Expected output
        expected_infections = np.zeros((N, kmax))
        expected_noninfections = np.zeros((N, kmax))
        # expected_noninfections[:, 0] = [99]*N

        # Compute activations
        actual_infections, actual_noninfections = src.dynamics.get_activations(states, adjacency, kmax)

        logger.debug(f"Infection events:\n {actual_infections}")
        logger.debug(f"Non-infection events:\n {actual_noninfections}")

        assert np.equal(actual_infections, expected_infections).all()
        assert np.equal(actual_noninfections, expected_noninfections).all()

    # @pytest.mark.skip()
    def test_perfect_contagion_small_ring(self):
        # Simple topology
        adjacency = np.array(
            [
                [0, 1, 0, 0, 1],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1],
                [1, 0, 0, 1, 0],
            ],
            dtype=int
        )
        kmax = 2

        # Perfect contagion
        states = np.array(
            [
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 1, 1, 1, 1],
            ],
            dtype=int
        )

        # Expected outcomes
        expected_infections = np.array(
            [
                [0, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
            ],
            dtype=int
        )
        expected_noninfections = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            dtype=int
        )

        # Compute activations
        actual_infections, actual_noninfections = src.dynamics.get_activations(states, adjacency, kmax)
        actual_infections = np.array(actual_infections, dtype=int)
        actual_noninfections = np.array(actual_noninfections, dtype=int)

        logger.debug(f"Infection events:\n {actual_infections}")
        logger.debug(f"Non-infection events:\n {actual_noninfections}")

        # assert np.equal(actual_infections, expected_infections).all()
        assert np.equal(actual_noninfections, expected_noninfections).all()