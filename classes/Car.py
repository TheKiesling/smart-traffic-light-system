class Car:
    """
    A car object to represent a car in the simulation.
    
    Args:
        id (str): The car's unique identifier.
        tca (TCA): The traffic cellular automaton where the car is located.
        max_speed (int): The car's maximum speed.
        direction (int): The direction of the car [0-359 degrees]. *
        status (int): The current status of the car (e.g., 0: stopped, 1: moving, 2: waiting). *
        length (int): The length of the car.
        driver (Driver): The driver controlling the car.
        speed (int, optional): The car's current speed. Defaults to 0. *
    """
    def __init__(self, id, tca, max_speed, direction, status, length, driver, speed=0):
        self.id = id
        self.tca = tca
        self.speed = speed
        self.max_speed = max_speed
        self.direction = direction
        self.status = status
        self.length = length
        self.driver = driver

    def accelerate(self, acceleration):
        """
        Increase the car's speed by its acceleration rate,
        ensuring it does not exceed the maximum speed.
        
        Args:
            acceleration (int): The acceleration rate of the car.
        """
        new_speed = self.speed + acceleration
        self.speed = min(new_speed, self.max_speed)
        self.status = 1
        print(f"Car {self.id} accelerated to {self.speed}.")

    def decelerate(self, deceleration):
        """
        Decrease the car's speed by its deceleration rate,
        ensuring the speed does not drop below 0.
        
        Args:
            deceleration (int): The deceleration rate of the car.
        """
        new_speed = self.speed + deceleration
        self.speed = max(new_speed, 0)
        if self.speed == 0:
            self.status = 2
        print(f"Car {self.id} decelerated to {self.speed}.")

    def change_direction(self, new_direction):
        """
        Change the direction of the car.
        
        Args:
            new_direction (int): New direction in degrees (0-359).
        """
        self.direction = new_direction % 360
        print(f"Car {self.id} changed direction to {self.direction} degrees.")

    def update_status(self, new_status):
        """
        Update the status of the car.
        
        Args:
            new_status (int): New status code.
        """
        self.status = new_status
        print(f"Car {self.id} status updated to {self.status}.")

    def __str__(self):
        return f"Car {self.id}: Speed={self.speed}, Direction={self.direction}, Status={self.status}"