import os
import time

GPIO = 138
FREQ_HZ = 25
UPDATE_S = 5
KICKSTART_S = 0.5
PERIOD_US = int(1_000_000 / FREQ_HZ)

T_FULL = 40.0
DUTY_MID = 80
DUTY_FULL = 100

GPIO_DIR = f"/sys/class/gpio/gpio{GPIO}"

def sh(cmd: str) -> str:
    return os.popen(cmd).read().strip()

def ensure_gpio():
    if not os.path.isdir(GPIO_DIR):
        sh(f"echo {GPIO} > /sys/class/gpio/export 2>/dev/null || true")
    sh(f"echo out > {GPIO_DIR}/direction")

def write_gpio(val: int):
    with open(f"{GPIO_DIR}/value", "w") as f:
        f.write("1" if val else "0")

def temp_c() -> float:
    return int(sh("cat /sys/class/thermal/thermal_zone0/temp")) / 1000.0

def duty_from_temp(t: float) -> int:
    return DUTY_FULL if t >= T_FULL else DUTY_MID

def pwm_run(duty: int, seconds: float):
    if duty >= 100:
        write_gpio(1)
        time.sleep(seconds)
        return

    on_us = int(PERIOD_US * duty / 100)
    off_us = PERIOD_US - on_us
    end = time.time() + seconds
    while time.time() < end:
        write_gpio(1); time.sleep(on_us / 1_000_000)
        write_gpio(0); time.sleep(off_us / 1_000_000)

def main():
    ensure_gpio()
    last_duty = 0

    while True:
        t = temp_c()
        duty = duty_from_temp(t)

        if last_duty == 0 and duty > 0:
            write_gpio(1)
            time.sleep(KICKSTART_S)

        print(f"temp: {t:.1f}C  duty: {duty}%")
        pwm_run(duty, UPDATE_S)
        last_duty = duty

if __name__ == "__main__":
    main()
