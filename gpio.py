import RPi.GPIO as GPIO
from apscheduler.schedulers.blocking import BlockingScheduler

sched = BlockingScheduler()

GPIO.setmode(GPIO.BCM)
GPIO.setup(2, GPIO.IN)


@sched.scheduled_job('interval', seconds=1)
def my_job():
    if GPIO.input(2):
        print("some one")
    else:
        print("no one")


sched.start()
