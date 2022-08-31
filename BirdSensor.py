from argparse import MetavarTypeHelpFormatter
from distutils.log import info
from multiprocessing.connection import wait
import time
import pandas as pd
from typing import List
import weakref
import json
import numpy as np
import pickle
import asyncio
import random
import logging
import ebird.api as bird
import requests
import os
import MLEngine as ml
from datetime import datetime, timedelta

#Make a new log file each time we run simulation. Paramerize log file name with a datetime value
LOG_TIME_FILE = time.strftime("%m%d%y%H%M%S", time.localtime())
logging.basicConfig(filename="C:\\Users\\thoma\\Desktop\\IOT_FINAL\\Simulation\\{}.txt".format(LOG_TIME_FILE),format="%(message)s", level=logging.INFO)

BIRD_PHOTO_DIR = "C:\\Users\\thoma\\Desktop\\IOT_FINAL\\ML\\WI_MODEL\\test\\"
SAMPLE_PHOTOS =[]
for path, currentDirectory, files in os.walk(BIRD_PHOTO_DIR):
    for file in files:
        SAMPLE_PHOTOS.append(os.path.join(path, file))
with open("C:\\Users\\thoma\\Desktop\\IOT_FINAL\\Simulation\\birdPopulationNames.txt",'rb') as f:
   BIRD_POPULATION_DATA = list(pickle.load(f))

WI_SPECIES = {'NORTHERN FLICKER': 'norfli',
 'CEDAR WAXWING': 'cedwax',
 'HOUSE FINCH': 'houfin',
 'HOUSE SPARROW': 'houspa',
 'DOWNY WOODPECKER': 'dowwoo'}

MERLIN_API_KEY = ""
WEATHER_API_KEY = ""

MY_MODEL = ml.MLEngine()
def main():
    mySim = simulation(engine_in=MY_MODEL)
    mySim.run_sim(mode='photo')

class simulation:
    def __init__(self, locationsFile="C:\\Users\\thoma\\Desktop\\IOT_FINAL\\Simulation\\locationData.txt", birdsFile = '',engine_in='') -> None:
        #List of the devices we're going to use for our run of the simulation
        with open(locationsFile,'rb') as f:
            self.locationsData = json.load(f)
        self.devices = []
        self.locations = []
        self.engine=engine_in
        #location of log file we're going to use
        self.logfile = locationsFile
        self.birdPopulation=[]
        self.fill_locations()
    def __str__(self):
        for location in self.locations:
            print("Research area: {}".format(str(location)))
    def run_sim(self, duration=300, frequency=0.5, mode='text'):
        startTime = time.time()
        if mode == 'text':
            while time.time() < startTime+duration:
                #for v1, we're just using random module to pick a bird name from a list
                bird = random.choice(BIRD_POPULATION_DATA)
                # Pick a random location
                location = random.choice(self.locations)
                #pick a random sensor from our location
                sensor = location.getRandomSensor()
                #Send the random bird to our random sensor
                sensor.reportBird(bird)
                time.sleep(frequency)
        if mode == 'photo':
            while time.time() < startTime+duration:
                #for v1, we're just using random module to pick a bird name from a list
                bird = random.choice(SAMPLE_PHOTOS)
                # Pick a random location
                location = random.choice(self.locations)
                #pick a random sensor from our location
                sensor = location.getRandomSensor()
                #Send the random bird to our random sensor
                sensor.birdVisit(birdPhoto=bird)
                time.sleep(frequency)
            '''
            FOR V2 - There are a few main changes that will need to be made. 
            1) simulator
            will select a random image file instead of just a random string from a list
            2) instead of just calling "reportBird" using the bird type, I need to pass that image to the appropriate function for ML call, APIs, and logging
            '''
    def generate_sensors(self):
        for location in self.locationsData:
            self.locations.append(location['name'])
            lats = np.linspace(location['minLat'],location['maxLat'],location['density'])
            longs = np.linspace(location['minLong'],location['maxLong'],location['density'])
            coordinates = np.array((lats,longs)).T
            locationDeviceCount=0
            for coordinate in coordinates:
                self.devices.append(BirdSensor(area=location['name'],location=coordinate,name=location['name']+str(locationDeviceCount)))
                locationDeviceCount += 1
    def fill_locations(self):
        for area in self.locationsData:
            self.locations.append(location(area))

class BirdSensor:
    def __init__(self, area = '', location =[], name='',mlEngine='') -> None:
        #GPS LOCATION
        self.lat=location[0]
        self.long=location[1]
        self.area=area
        self.name=name
        self.logFile =''
        self.engine=mlEngine
    def __str__(self):
        return("Device: {} at {}, {}".format(self.name, self.lat,self.long))
    def birdVisit(self,birdPhoto):
        birdName, observationID = self.inferBird(birdPhoto)
        weather = self.getWeatherData()
        #get a random time in next week or so to say that the observation happened at
        timeofVisit = datetime.now() + timedelta(days=random.randint(1,4), seconds=random.randint(100,900))
        currentVisitData ={
            'comName': birdName,
            'speciesCode': WI_SPECIES[birdName],
            'obsDt':timeofVisit.strftime("%G-%m-%d %H:%M"),
            'lat': self.lat,
            'lng': self.long,
            'observation':'realTime',
            'sensorName':self.name,
            'areaName':self.area,
            'observationID': observationID
        }
        #Created a combined dictionary of the inferred bird + data on the date/location of sensor
        currentBirdObs = currentVisitData|weather
        logging.info(json.dumps(currentBirdObs))
        #get the recent observations of our bird
        recentObservations = self.getMerlinData(birdName,observationID=observationID)
        #loop through returned list of dictionaries and write them to the log file
        for observation in recentObservations:
            logging.info(json.dumps(observation))
    def takePhoto():
        return 'photo'    
    def inferBird(self,birdPhoto):
        inferredBird=MY_MODEL.inferPhoto(birdPhoto)
        observationID = time.strftime("%H%M%S", time.localtime())
        return inferredBird, observationID
    def getMerlinData(self, visitingBird, observationID):
        speciesCode =WI_SPECIES[visitingBird]
        pastObserved = bird.get_nearest_species(token=MERLIN_API_KEY,species=speciesCode, lat=self.lat, lng=self.long, dist=25,back=5,max_results=5)
        recentSightings =[]
        for observe in pastObserved:
            recentSightings.append({
            'speciesCode': observe['speciesCode'],
            'comName': visitingBird,
            'obsDt': observe['obsDt'],
            'lat': observe['lat'],
            'lng': observe['lng'],
            'observation': 'Historical',
            'observationID':observationID
            })
        return recentSightings
    def getWeatherData(self):
        payload = {
        "lat":self.lat,
        "lon":self.long,
        "units":'imperial',
        "appid":WEATHER_API_KEY
        }
        weatherResponse = requests.get('https://api.openweathermap.org/data/2.5/weather?', params=payload).json()
        currentWeather = {
            'weatherDescription':weatherResponse['weather'][0]['main'],
            'temperature':weatherResponse['main']['temp'],
            'windSpeed':weatherResponse['wind']['speed'],
            'pressure':weatherResponse['main']['pressure']
        }
        return currentWeather
    def reportBird(self, visitingBird=''):
        birdEvent = {}
        #birdPhoto=self.takePhoto()
        #bird=self.inferPhoto(birdPhoto)
        #birdData=self.getMerlinData(bird)
        #weatherData=self.getWeatherData()
        logging.info('Just saw a {} at {}'.format(visitingBird, self.name))
        print('Just saw a {} at {}'.format(visitingBird, self.name))



class location:
    def __init__(self, locationData={}):
        self.area = locationData['name']
        self.devices = []
        self.deviceCount=0
        self.density = locationData['density']
        self.type = locationData['type']
        self.profile = locationData['profile']
        self.minLat =locationData['minLat']
        self.maxLat =locationData['maxLat']
        self.minLong =locationData['minLong']
        self.maxLong =locationData['maxLong']
        self.generate_sensors()
    def generate_sensors(self):
        lats = np.linspace(self.minLat,self.maxLat,self.density)
        longs = np.linspace(self.minLong,self.maxLong,self.density)
        coordinates = np.array((lats,longs)).T
        for coordinate in coordinates:
            self.devices.append(BirdSensor(area=self.area,location=coordinate,name=self.area+str(self.deviceCount)))
            self.deviceCount += 1
    def __str__(self):
        return "Location {} has {} sensors".format(self.area, str(self.deviceCount))
    def getRandomSensor(self):
        return random.choice(self.devices)



if __name__ == '__main__':
    main()