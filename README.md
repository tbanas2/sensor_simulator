# Summary
This program is proof of concept for a "smart" bird feeder I designed for a hardware class. The program accepts parameters to simulate a deployment of these feeders to
demonstrate the value of the data they could generate

## Background
Parameters for geographic location and number of sensors are accepted so that a scientific study in a specific area using the sensors can be simulated. When a bird visits a feeder,
the following happens:

1. The species is identified using a model I trained with PyTorch (image files are used in the simulator, but it would be a camera in real life)
2. A call is made to the eBird API to check for nearby sightings of the species to try and determine where the bird may have came from
3. A call is made to the weather API to get current weather conditions
4. The data is serialized and logged

The resulting data can be loaded into an analysis tool and used to answer a wide range of scientific questions: what birds visit the area when, what other nearby areas are 
important to them, etc.
