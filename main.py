import time
from classes.Car import Car
from classes.Driver import Driver


def main():
    class TCA:
        pass

    tca_instance = TCA()

    driver_aggressive = Driver(
        id="Agresivo",
        aggressiveness=0.9,
        reaction_time=500,
        risk_tolerance=0.8,
        lane_discipline=0.2,
        distraction_level=0.3,
        awareness=0.6,
        impulsivity=0.9
    )
    driver_average = Driver(
        id="Promedio",
        aggressiveness=0.4,
        reaction_time=600,
        risk_tolerance=0.3,
        lane_discipline=0.8,
        distraction_level=0.2,
        awareness=0.8,
        impulsivity=0.4
    )

    driver_distracted = Driver(
        id="Distraido",
        aggressiveness=0.2,
        reaction_time=900,
        risk_tolerance=0.2,
        lane_discipline=0.5,
        distraction_level=0.9,
        awareness=0.3,
        impulsivity=0.5
    )


    car1 = Car(id="Car1", tca=tca_instance, max_speed=120, 
               direction=0, status=0, length=4, driver=driver_aggressive)
    car2 = Car(id="Car2", tca=tca_instance, max_speed=100, 
               direction=90, status=0, length=4, driver=driver_average)
    car3 = Car(id="Car3", tca=tca_instance, max_speed=80, 
               direction=180, status=0, length=4, driver=driver_distracted)

    vehicles = [car1, car2, car3]

    epochs = 5

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}")
        for car in vehicles:
            car.driver.decide_action(car)
        print("\nStates after epoch:")
        for car in vehicles:
            print(car)
        time.sleep(1)


if __name__ == '__main__':
    main()

