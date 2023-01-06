# alice_satellite

[list of supported wordds in the model](tflite/labels.txt)

The goal of this project is to reduce the complexity and setup for Satellite clients in a Rhasspy enviroment.  There is no UI or web interface; and provides a yaml configuration file. A requirement is to work 100% offline without the need for an internet connection.

- This provides a framework to generate and train a new wake word, custom to your needs (ie any language) using Tensorflow; heavily pulls from google kws research project.
- Will listen for a wake word, and send audio bytes to a rhasspy server
- Will respond to PlayBytes requests from a rhasspy server

This is the initial prototype which I plan to use and maintain for my Home Assistant setup; your welcome to use it as well.

This makes some assumptions:
- You are using an MQTT broker
- You have Rhasspy setup in your enviroment, for handling audio requests
- HomeAssistant to handle all the intents

# Roadmap
As this project is just getting started; I will refine and update the documentation over the next few months.

# Demo

I've included a Dockerfile with a pre-trained model with the speech commands for testing out the interfaces and configurations.  You can even run this from any system with a microphone and speakers for testing and evaluation.

From the root of the checkout run the following to build an image:
```Shell
docker build . -t alice -f Docker/Dockerfile
docker run --name alice --device /dev/snd:/dev/snd -dit alice
docker attach alice

cd /root
./demo.sh
```

or you can just check it out and run: inside the Docker folder is a requirements.txt for the dependencies. You will need to copy the tflite into your data folder {alice_data/tflite}
```shell
python -m pip install 'alice_satellite @ git+https://github.com/UmbrellaCodr/alice_satellite@main'
```
```Shell
python -m alice_satellite -v -h
```

# supported commands
```Shell
python -m alice_satellite -h
usage: alice [-h] [--data DATA] [-d] [-v] [--audio_input AUDIO_INPUT] [--audio_output AUDIO_OUTPUT]
             {list,config,analyze,listen,detect,satellite,train,predict,generate,morph,verify,info,mqtt,tts,transcribe} ...

positional arguments:
  {list,config,analyze,listen,detect,satellite,train,predict,generate,morph,verify,info,mqtt,tts,transcribe}
                        supported sub commands
    list                list audio devices
    config              save a config file
    analyze             morph samples
    listen              requires a trained model
    detect              requires a trained model
    satellite           main mode, listens for wake word communicates to rhasspy
    train               train the model
    predict             classify wav file
    generate            generate samples
    morph               morph samples
    verify              verify samples
    info                dump data folder
    mqtt                mqtt util
    tts                 text to speech
    transcribe          transcribe audio file

options:
  -h, --help            show this help message and exit
  --data DATA           set the default data location
  -d, --debug           maybe print something helpful
  -v, --verbose         log more
  --audio_input AUDIO_INPUT
                        specify in device from {list}
  --audio_output AUDIO_OUTPUT
                        specify out device from {list}
```

{alice_data} defaults to the current location where you are invoking the model from unless specified by --data

- **list**
Will show all the in/out devices currently detected by the application
- **config**
This will generate a config.yml into your {alice_data} folder 
- **analyze**
This uses whisper to validate generated samples match an alternative machine model
- **listen**
This will process input from the microphone and compare it against the current model
- **detect**
This is for debugging and validating detecting of a keyword
- **satellite**
This is the main mode will describe down bellow
- **train**
this is used to train a new tensorflow model with any generates samples
- **generate**
this is a mode to generate and prepare samples for new keywords used by the train command
- **morph**
takes your samples and shifts the audio around in the window for training
- **verify**
allows you to listen to your recorded samples
- **info**
Will show you a summary of the current tflite model and samples 
- **mqtt**
This allows you to subscribe and listen to mqtt topics on the network mostly used for debugging
- **tts**
this allow you to pass a string and get an audio file back from Rhasspy
- **transcribe**
this allows you to pass an audio file to Rhasspy to see if it detected an intent

# Satellite
Satellite mode allows you to setup an array of required words or a single word based on the parameters passed

This will indicate that both 2 and 3 need to be spoken within a 3 second window before having the wake word detected, for example "Hey Alice" where Hey is at index 2 and Alice is at index 3
```Shell
-i 2 3
```

You can also pass -m 15 where if index 15 is matched it will enable wake word detection. You also have the ability to adjust the threshold by passing --threshold 
```Shell
python -m alice_satellite -v satellite -h
usage: alice satellite [-h] [-i INDEX [INDEX ...]] [-m MATCH] [-t THRESHOLD] [-w]

options:
  -h, --help            show this help message and exit
  -i INDEX [INDEX ...], --index INDEX [INDEX ...]
                        index of words to required for wake word detection
  -m MATCH, --match MATCH
                        single index to match for wake word detection
  -t THRESHOLD, --threshold THRESHOLD
                        threshold for keyword match
  -w, --whisper         enable whisper transcription
```
# screenshots
<img width="1042" alt="image" src="https://user-images.githubusercontent.com/95475404/210164741-577bcd7b-54a2-42fc-ae51-b30047d29973.png">

```python
. = no noise detected
* = noise detected
* yellow = one of the keywords detected under threshold
* green = one of the keywords detected
detected = when both keywords were presented in a 3 second window
+ = it is streaming audio to Rhasspy
P = Rhasspy sent us a sound to play
S = Rhasspy sent us a start/stop listening request
```

<img width="1037" alt="image" src="https://user-images.githubusercontent.com/95475404/210164821-6d352a1d-fdb3-4368-92dd-8464209c14f3.png">

# mqtt settings

To configure the mqtt settings you need a configuration file you can create a default one by running the config command. {alice_data/config.yml}

We support all of the following mqtt settings here: [mqtt settings](https://sbtinstruments.github.io/asyncio-mqtt/configuring-the-client.html)

### The only non-optional parameter
```yaml
mqtt:
  hostname: test.mosquitto.org
```

Example:
```yaml
mqtt:
  hostname: test.mosquitto.org
  password: password
  port: 1883
  username: alice
```

# pi configuration
```
sudo vim /lib/systemd/system/alice.service
```
```
[Unit]
Description=Alice Satellite
After=multi-user.target

[Service]
Type=simple
User=pi
ExecStart=/usr/bin/python3 -m alice_satellite --data /home/pi/alice_data satellite -i 2 3 -m 4
Restart=on-abort

[Install]
WantedBy=multi-user.target
```
```
sudo chmod 644 /lib/systemd/system/alice.service
sudo systemctl daemon-reload
sudo systemctl enable alice.service
sudo systemctl start alice.service
```

## Service Tasks
For every change that we do on the /lib/systemd/system folder we need to execute a daemon-reload (third line of previous code). If we want to check the status of our service, you can execute:
```
sudo systemctl status alice.service
```
### In general:

Check status
```
sudo systemctl status alice.service
```
Start service
```
sudo systemctl start alice.service
```
Stop service
```
sudo systemctl stop alice.service
```
Check service's log
```
sudo journalctl -f -u alice.service
```
