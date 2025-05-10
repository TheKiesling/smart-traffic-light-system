import random

class Driver:
    """
    A driver object to represent a driver in the simulation.
    
    Args:
        id (str): The driver's unique identifier.
        aggressiveness (float): Level of aggressiveness [0-1] (0: passive, 1: aggressive).
        reaction_time (int): Response time to events in milliseconds.
        risk_tolerance (float): How close the driver is willing to get to other cars [0-1] (0: low, 1: high).
        lane_discipline (float): Likelihood of the driver to change lanes [0-1] (0: low, 1: high).
        distraction_level (float): Likelihood of being slow to react [0-1] (0: low, 1: high).
        patience (float): Tolerance to delays [0-1] (0: low, 1: high).
        adaptability (float): Ability to adjust behavior according to the context [0-1] (0: low, 1: high).
        awareness (float): Level of perception of the environment [0-1] (0: low, 1: high).
    """
    def __init__(self, id, aggressiveness, reaction_time, risk_tolerance, lane_discipline, distraction_level, awareness, impulsivity):
        self.id = id
        self.aggressiveness = aggressiveness
        self.reaction_time = reaction_time
        self.risk_tolerance = risk_tolerance
        self.lane_discipline = lane_discipline
        self.distraction_level = distraction_level
        self.awareness = awareness
        self.impulsivity = impulsivity


    def decide_action(self, car):
        """
        Decide the action of the driver based on their attributes and the car's state.

        Args:
            car (Car): The car object controlled by the driver.
        """
        decision_factor = random.random()

        distraction_penalty = self.distraction_level * 0.30
        reaction_penalty = (self.reaction_time / 1000) * 0.20
        risk_bonus = self.risk_tolerance * 0.15
        lane_discipline_bonus = self.lane_discipline * 0.10
        awareness_bonus = self.awareness * 0.20
        aggressiveness_bonus = self.aggressiveness * 0.25
        impulsivity_bonus = self.impulsivity * 0.15 

        adjusted_factor = (
            decision_factor
            - distraction_penalty
            - reaction_penalty
            + risk_bonus
            + lane_discipline_bonus
            + awareness_bonus
            + aggressiveness_bonus
            + impulsivity_bonus
        )

        if adjusted_factor > 0.8 and car.speed < car.max_speed:
            print(f"Driver {self.id} decide acelerar agresivamente.")
            car.accelerate(decision_factor * 2.0)
        elif adjusted_factor > 0.6 and car.speed < car.max_speed:
            print(f"Driver {self.id} decide acelerar moderadamente.")
            car.accelerate(decision_factor * 1.0)
        elif adjusted_factor > 0.3 and car.speed > 0:
            print(f"Driver {self.id} mantiene velocidad actual.")
        elif car.speed > 0:
            print(f"Driver {self.id} decide desacelerar por factor bajo ajustado.")
            car.decelerate(decision_factor * -1.0)
        else:
            print(f"Driver {self.id} espera, no es seguro avanzar.")

    def __str__(self):
        return f"Driver {self.id}: Aggressiveness={self.aggressiveness}, Reaction Time={self.reaction_time}ms"