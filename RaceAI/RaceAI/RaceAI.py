# IMPORTS
import pygame
from random import randint
import math

# COLORS
white = (255, 255, 255)
black = (0, 0, 0)
grey = (200, 200, 200)
red = (255, 0, 0)

# CAR SETTINGS
START_POS = [425, 485]
maxSpeed = 4.5
acceleration = 0.10
deacceleration = 0.06
drag = 0.05
turnSpeed = 3

# WINDOW SETTINGS
pygame.init()
(width, height) = (800, 600)
GD = pygame.display.set_mode((width, height))
pygame.display.set_caption('Race AI')
GD.fill(white)
clock = pygame.time.Clock()
cooldown = 6 # amount of frames before the next collision check
cooldownid = -1 # keeps track of the passed frames

# IMAGES
carimage = pygame.transform.scale(pygame.image.load(r"images\greencar.png"), (40, 22))
track = pygame.transform.scale(pygame.image.load(r"images\racetrack.png"), (800, 600))
greentrack = pygame.transform.scale(pygame.image.load(r"images\mooibaan.png"), (800, 600))
finishline = pygame.transform.scale(pygame.image.load(r"images\finishline.png"), (7, 100))
checkpoint = pygame.transform.scale(pygame.image.load(r"images\checkpoints.png"), (800, 600))

# AI SETTINGS
GENERATIONSIZE = 20
currentGen = []
bestFitness = 0
eliteFitness = 0

# NEURAL NETWORK SETTINGS
inputSpread = 20
hiddenSpread = 20
biasSpread = 50

INPUTNEURONS = 5
HIDDENNEURONS = 3
OUTPUTNEURONS = 3
activation = [0.0, 0.0, 0.0]

# BEST BRAIN
bestinputweights = []
besthiddenweights = []
besthiddenbias = []
bestoutputbias = []

# ELITE BRAIN
eliteinputweights = []
elitehiddenweights = []
elitehiddenbias = []
eliteoutputbias = []

# VISUALISATION SETTINGS
NETWORKPOSITION = [470, 40]
NEURONSIZE = 20
NEURONOUTLINE = 2
NEURONSPACING = 50
LAYERSPACING = 120

# KEYS
# w - 1, s - 0, a - 3, d - 2

def softmax(outputs):
    # Find the maximum value to stabilize the calculation and avoid overflow
    max_output = max(outputs)
    
    # Calculate exponentials for each output after subtracting max_output
    exp_values = [math.exp(output - max_output) for output in outputs]
    
    # Calculate the sum of all exponentials
    sum_exp_values = sum(exp_values)
    
    # Divide each exponential by the sum to get probabilities
    probabilities = [exp_val / sum_exp_values for exp_val in exp_values]
    return probabilities

def reset():
    # Deze variabelen worden global benoemd hierzo zodat de code de bestaande variabelen gebruikt en niet nieuwe aan maakt.
    global bestFitness
    global eliteFitness
    global bestinputweights
    global besthiddenweights
    global besthiddenbias
    global bestoutputbias
    global eliteinputweights
    global elitehiddenweights
    global elitehiddenbias
    global eliteoutputbias
    bestFitness = 0

    for i in range(len(currentGen)):
        # Als de huidige fitness beter is dan de beste fitness van deze generatie dan slaan we zijn neural network op.
        if currentGen[i].fitness > bestFitness:
            bestFitness = currentGen[i].fitness
            bestinputweights = currentGen[i].inputweights
            besthiddenweights = currentGen[i].hiddenweights
            besthiddenbias = currentGen[i].hiddenbias
            bestoutputbias = currentGen[i].outputbias
        # Als de huidige fitness beter is dan de beste fitness ooit dan slaan we zijn neural network op.
        if currentGen[i].fitness >= eliteFitness:
            eliteFitness = currentGen[i].fitness
            eliteinputweights = currentGen[i].inputweights
            elitehiddenweights = currentGen[i].hiddenweights
            elitehiddenbias = currentGen[i].hiddenbias
            eliteoutputbias = currentGen[i].outputbias
    
    # Als de beste fitness van deze generatie (best) slechter was dan de beste ooit, dan pakken we de beste neural network (elite) voor de volgende generatie.
    # Dit doen we om te vermijden dat het neural network zich achteruit ontwikkelt
    if bestFitness < eliteFitness:
        bestinputweights = eliteinputweights
        besthiddenweights = elitehiddenweights
        besthiddenbias = elitehiddenbias
        bestoutputbias = eliteoutputbias
    
    # Hier printen we na elke generatie de fitnesses
    print("Best fitness: " + str(bestFitness))
    print("Elite fitness: " + str(eliteFitness))

    # Alle autos worden hiermee verwijderd en de lijst voor vrij gemaakt voor de nieuwe generatie.
    currentGen.clear()

    # Hier maken we een nieuwe generatie aan.
    spawnGen()
    if bestFitness == 0:
        for i in range(len(currentGen)):
            # Als alle autos geen fitness gehaald hebben, dan genereren we weer compleet willekeurige neural networks.
            currentGen[i].setupNetwork()
    else:
        for i in range(len(currentGen)):
            # Als de autos wel fitness gehaald hebben, geven we de nieuwe generatie allemaal het beste brein van de vorige generatie en maken we kleine aanpassingen per auto.
            currentGen[i].modifyNetwork()

# Hier maken we een nieuwe generatie aan
def spawnGen():
    for i in range(GENERATIONSIZE):
        currentGen.append(Car(carimage, START_POS[0], START_POS[1], 0, 0, 180, [], [], [], bestinputweights.copy(), besthiddenweights.copy(), besthiddenbias.copy(), bestoutputbias.copy(), [], False, False, 0))

class Car():
    # In deze class wordt de auto gedefineerd. Self zorgt ervoor dat hij alles van de eigen auto pakt i.p.v een andere auto
    def __init__(self, image, x, y, speed, fitness, rotation, inputs, hidden, output, inputweights, hiddenweights, hiddenbias, outputbias, KEYS, dead, cooldown, countdown):
        self.image = image
        self.x = x
        self.y = y
        self.speed = speed
        self.fitness = fitness
        self.angle = 0 
        self.rect = self.image.get_rect(center=(self.x, self.y)) 
        self.position = pygame.Vector2(self.x, self.y)
        self.rotation = rotation
        self.inputs = inputs
        self.hidden = hidden
        self.output = output
        self.inputweights = inputweights
        self.hiddenweights = hiddenweights
        self.hiddenbias = hiddenbias
        self.outputbias = outputbias
        self.KEYS = KEYS
        self.dead = dead
        self.cooldown = cooldown
        self.countdown = countdown

    # Rotate
    @property
    def forward(self):
        # Convert angle to radians
        radians = math.radians(self.rotation)
        # Calculate forward direction
        return pygame.Vector2(math.cos(radians), math.sin(radians))

    @property
    def left(self):
        # Rotate forward vector by -90 degrees (clockwise)
        return pygame.Vector2(self.forward.y, -self.forward.x)

    @property
    def right(self):
        # Rotate forward vector by 90 degrees (counter-clockwise)
        return pygame.Vector2(-self.forward.y, self.forward.x)
    
    @property
    def left_forward(self):
        # Rotate forward vector by -45 degrees (clockwise)
        radians = math.radians(self.rotation - 45)
        return pygame.Vector2(math.cos(radians), math.sin(radians))

    @property
    def right_forward(self):
        # Rotate forward vector by 45 degrees (counter-clockwise)
        radians = math.radians(self.rotation + 45)
        return pygame.Vector2(math.cos(radians), math.sin(radians)) 

    # Meten hoe ver de auto van de kant af zit.
    def position_to_front(self, distance):
        return self.position + self.forward * distance

    def position_to_left(self, distance):
        return self.position + self.left * distance

    def position_to_right(self, distance):
        return self.position + self.right * distance
    
    def position_to_forward_left(self, distance):
        return self.position + self.left_forward * distance

    def position_to_forward_right(self, distance):
        return self.position + self.right_forward * distance

    # Uitrekenen hoever de auto van de kant af zit. De auto berekent dit in stappen van 10 pixels zodat de computer niet teveel berekeningen hoeft te doen.
    def distanceToFront(self):
        pos = 0
        for i in range(60):
            pos = self.position_to_front(10+(i*10))
            if (pos.x < 0 or pos.x > width or pos.y < 0 or pos.y > height):
                for j in range(10):
                    pos = self.position_to_front((10+((i-1)*10))-(j+1))
                    if GD.get_at((int(pos.x), int(pos.y))) == black:
                        self.inputs[0] = (10+((i-1)*10))-(j+1)
                        return
            else:
                if GD.get_at((int(pos.x), int(pos.y))) == black:
                    for j in range(10):
                        pos = self.position_to_front((10+(i*10))-(j+1))
                        if GD.get_at((int(pos.x), int(pos.y))) != black:
                            self.inputs[0] = (10+(i*10))-(j+1)
                            return

    def distanceToLeft(self):
        pos = 0
        for i in range(60):
            pos = self.position_to_left(10+(i*10))
            if (pos.x < 0 or pos.x > width or pos.y < 0 or pos.y > height):
                for j in range(10):
                    pos = self.position_to_left((10+((i-1)*10))-(j+1))
                    if GD.get_at((int(pos.x), int(pos.y))) == black:
                        self.inputs[1] = (10+((i-1)*10))-(j+1)
                        return
            else:
                if GD.get_at((int(pos.x), int(pos.y))) == black:
                    for j in range(10):
                        pos = self.position_to_left((10+(i*10))-(j+1))
                        if GD.get_at((int(pos.x), int(pos.y))) != black:
                            self.inputs[1] = (10+(i*10))-(j+1)
                            return
    
    def distanceToRight(self):
        pos = 0
        for i in range(60):
            pos = self.position_to_right(10+(i*10))
            if (pos.x < 0 or pos.x > width or pos.y < 0 or pos.y > height):
                for j in range(10):
                    pos = self.position_to_right((10+((i-1)*10))-(j+1))
                    if GD.get_at((int(pos.x), int(pos.y))) == black:
                        self.inputs[2] = (10+((i-1)*10))-(j+1)
                        return
            else:
                if GD.get_at((int(pos.x), int(pos.y))) == black:
                    for j in range(10):
                        pos = self.position_to_right((10+(i*10))-(j+1))
                        if GD.get_at((int(pos.x), int(pos.y))) != black:
                            self.inputs[2] = (10+(i*10))-(j+1)
                            return

    def distanceToForwardLeft(self):
        pos = 0
        for i in range(60):
            pos = self.position_to_forward_left(10+(i*10))
            if (pos.x < 0 or pos.x > width or pos.y < 0 or pos.y > height):
                for j in range(10):
                    pos = self.position_to_forward_left((10+((i-1)*10))-(j+1))
                    if GD.get_at((int(pos.x), int(pos.y))) == black:
                        self.inputs[3] = (10+((i-1)*10))-(j+1)
                        return
            else:
                if GD.get_at((int(pos.x), int(pos.y))) == black:
                    for j in range(10):
                        pos = self.position_to_forward_left((10+(i*10))-(j+1))
                        if GD.get_at((int(pos.x), int(pos.y))) != black:
                            self.inputs[3] = (10+(i*10))-(j+1)
                            return
                        
    def distanceToForwardRight(self):
        pos = 0
        for i in range(60):
            pos = self.position_to_forward_right(10+(i*10))
            if (pos.x < 0 or pos.x > width or pos.y < 0 or pos.y > height):
                for j in range(10):
                    pos = self.position_to_forward_right((10+((i-1)*10))-(j+1))
                    if GD.get_at((int(pos.x), int(pos.y))) == black:
                        self.inputs[4] = (10+((i-1)*10))-(j+1)
                        return
            else:
                if GD.get_at((int(pos.x), int(pos.y))) == black:
                    for j in range(10):
                        pos = self.position_to_forward_right((10+(i*10))-(j+1))
                        if GD.get_at((int(pos.x), int(pos.y))) != black:
                            self.inputs[4] = (10+(i*10))-(j+1)
                            return

    # Self zorgt ervoor dat hij alles van de eigen auto pakt i.p.v een andere auto.
    def getInput(self):
        self.distanceToFront()
        self.distanceToLeft()
        self.distanceToRight()
        self.distanceToForwardLeft()
        self.distanceToForwardRight()

    # Hier wordt gedefinieerd hoe snel de auto accelereert en draait
    def move(self):
        if 10 in self.KEYS:
            self.speed += deacceleration
        if 0 in self.KEYS:
            self.speed -= acceleration
   
        if self.speed != 0:
            if 1 in self.KEYS:  
                if self.speed > 0:
                    self.angle += turnSpeed  
                else:
                    self.angle -= turnSpeed 
           
            if 2 in self.KEYS:  
                if self.speed > 0:
                    self.angle -= turnSpeed 
                else:
                    self.angle += turnSpeed  

        #CALC ANGLE
        radians = math.radians(self.angle)
        self.x += self.speed * math.cos(radians)
        self.y += self.speed * math.sin(radians)

        if self.speed > 0:
            if self.speed > maxSpeed:
                self.speed = maxSpeed
            if self.speed <= drag:
                self.speed = 0
            else:
                self.speed -= drag

        elif self.speed < 0:
            if self.speed < -maxSpeed:
                self.speed = -maxSpeed
            self.speed += drag

        # UPDATE LOCATION
        self.rect = self.image.get_rect(center=(self.x, self.y))
        self.position = pygame.Vector2(self.x, self.y)
        self.rotation = self.angle + 180

    def draw(self):
        # ROTATE SPRITE
        rotated_image = pygame.transform.rotate(self.image, -self.angle)  # Draai tegen de klok in
        rotated_rect = rotated_image.get_rect(center=self.rect.center)  # Zorg dat het draait om het midden
        GD.blit(rotated_image, rotated_rect.topleft)

    # Als de auto op een zwart gedeelte komt dan verdwijnt de auto 
    def checkCollision(self):
        if GD.get_at((int(self.x), int(self.y))) == black:
            #print("collision")
            self.dead = True
            self.speed = 0
            self.KEYS = []
    
    # Raakt de auto de finish dan krijg je 3 punten
    def checkRewardCollision(self):
        # Check finish collision
        if GD.get_at((int(self.x), int(self.y))) == red:
            if not self.cooldown:
                self.fitness += 3
            self.cooldown = True
        # Check checkpoint collision. De auto krijgt een punt wanneer de auto een grijze checkpoint in rijdt.
        elif GD.get_at((int(self.x), int(self.y))) == grey:
            if not self.cooldown:
                self.fitness += 1
            self.cooldown = True
        else:
            self.cooldown = False
    
    # Hier maakt het een nieuwe neural network aan voor een auto.
    def setupNetwork(self):
        for i in range(INPUTNEURONS):
            # Hier worden de input neurons aangemaakt.
            self.inputs.append(0)
            for j in range(HIDDENNEURONS):
                # Hier maakt de code voor elke hidden neuron per input neuron een nieuwe weight aan.
                self.inputweights.append(randint(-120, 120)/100)
        for i in range(HIDDENNEURONS):
            # Hier worden de hidden neurons aangemaakt.
            self.hidden.append(0)
            # Hier worden de hidden biases aangemaakt.
            self.hiddenbias.append(randint(-50, 50))
            for j in range(OUTPUTNEURONS):
                # Hier maakt de code voor elke output neuron per hidden neuron een nieuwe weight aan.
                self.hiddenweights.append(randint(-110, 110)/100)
        for i in range(OUTPUTNEURONS):
            # Hier worden de output neurons aangemaakt.
            self.output.append(0)
            # Hier worden de output biases aangemaakt.
            self.outputbias.append(randint(-50, 50))

    # Hier wordt het neural network aangepast, zodat hij een klein beetje anders is.
    def modifyNetwork(self):
        for i in range(INPUTNEURONS):
            # Hier worden de input neurons aangemaakt.
            self.inputs.append(0)
            for j in range(HIDDENNEURONS):
                # Hier wordt elke input weight een klein beetje aangepast.
                self.inputweights[(i*HIDDENNEURONS) + j] += (randint(-inputSpread, inputSpread)/100)
        for i in range(HIDDENNEURONS):
            # Hier worden de hidden neurons aangemaakt.
            self.hidden.append(0)
            # Hier wordt elke hidden bias aangepast.
            self.hiddenbias[i] += randint(-biasSpread, biasSpread)
            for j in range(OUTPUTNEURONS):
                # Hier wordt elke hidden weight een klein beetje aangepast.
                self.hiddenweights[(i*OUTPUTNEURONS) + j] += (randint(-hiddenSpread, hiddenSpread)/100)
        for i in range(OUTPUTNEURONS):
            # Hier worden de output neurons aangemaakt.
            self.output.append(0)
            # Hier wordt elke output bias aangepast.
            self.outputbias[i] += randint(-biasSpread, biasSpread)

    # Het neural netwerk wordt hier gerunned.
    def runNetwork(self):
        for h in range(HIDDENNEURONS):
            # Hier maken we elke hidden neuron zijn bias.
            self.hidden[h] = self.hiddenbias[h]
            for i in range(INPUTNEURONS):
                # Hier wordt elke input neuron vermenigvuldigd met zijn weight en toegevoegd aan de hidden neuron.
                self.hidden[h] += self.inputs[i] * self.inputweights[i]
                self.hidden[h] = max(0, self.hidden[h])
        for o in range(OUTPUTNEURONS):
            # Hier maken we elke output neuron zijn bias.
            self.output[o] = self.outputbias[o]
            for h in range(HIDDENNEURONS):
                # Hier wordt elke hidden neuron vermenigvuldigd met zijn weight en toegevoegd aan de output neuron.
                self.output[o] += self.hidden[h] * self.hiddenweights[h]
                self.output[o] = max(0, self.output[o])
                # Hier wordt elke output vereenvoudigd naar een getal tussen 0 en 1.
                probabilities = softmax(self.output)
                self.output = probabilities

        # Als een output neuron boven zijn activatie waarde is drukt hij zijn bijbehorende toets in.
        for o in range(OUTPUTNEURONS):
            if self.output[o] > activation[o]:
                if o == 0:
                    self.countdown = 0
                if not o in self.KEYS:
                    self.KEYS.append(o)
            else:
                self.countdown += 1
                if o in self.KEYS:
                    # Als een output neuron lager dan zijn activatie waarde is en zijn toets is ingedrukt dan wordt deze losgelaten.
                    self.KEYS.remove(o)
        # Als de countdown meer dan 30 is dan verwijderen we de auto.
        if self.countdown > 30:
            self.dead = True
        
        # Als een auto stil staat tikt de countdown omhoog.
        if self.speed == 0:
            self.countdown += 1
        else:
            self.countdown == 0

        # Om te vermijden dat links en rechts tegelijkertijd ingedrukt worden kijken we welke output waarde hoger is en drukken we alleen die toets in.
        if 1 in self.KEYS and 2 in self.KEYS:
            if self.output[1] > self.output[2]:
                self.KEYS.remove(2)
            else:
                self.KEYS.remove(1)

    # Hier tekenen we het neural network van de beste auto. Dit heeft geen functie buiten het visuele aspect.
    def drawNetwork(self):        
        for i in range(INPUTNEURONS):
            for h in range(HIDDENNEURONS):
                if self.inputweights[(i*HIDDENNEURONS)+h] > 0:
                    pygame.draw.line(GD, white, (NETWORKPOSITION[0], NETWORKPOSITION[1] + (i*NEURONSPACING)), (NETWORKPOSITION[0] + LAYERSPACING, NETWORKPOSITION[1] + ((h+1)*NEURONSPACING)))
                else:
                    pygame.draw.line(GD, black, (NETWORKPOSITION[0], NETWORKPOSITION[1] + (i*NEURONSPACING)), (NETWORKPOSITION[0] + LAYERSPACING, NETWORKPOSITION[1] + ((h+1)*NEURONSPACING)))

        for h in range(HIDDENNEURONS):
            for o in range(OUTPUTNEURONS):
                if self.hiddenweights[(h*OUTPUTNEURONS)+o] > 0:
                    pygame.draw.line(GD, white, (NETWORKPOSITION[0] + LAYERSPACING, NETWORKPOSITION[1] + ((h+1)*NEURONSPACING)), (NETWORKPOSITION[0] + LAYERSPACING * 2, NETWORKPOSITION[1] + ((o+1)*NEURONSPACING)))
                else:
                    pygame.draw.line(GD, black, (NETWORKPOSITION[0] + LAYERSPACING, NETWORKPOSITION[1] + ((h+1)*NEURONSPACING)), (NETWORKPOSITION[0] + LAYERSPACING * 2, NETWORKPOSITION[1] + ((o+1)*NEURONSPACING)))


        for i in range(INPUTNEURONS):
            pygame.draw.circle(GD, black, (NETWORKPOSITION[0], NETWORKPOSITION[1] + (i*NEURONSPACING)), NEURONSIZE)
            color = max(0, min(self.inputs[i], 255))
            pygame.draw.circle(GD, (color, color, color), (NETWORKPOSITION[0], NETWORKPOSITION[1] + (i*NEURONSPACING)), NEURONSIZE-NEURONOUTLINE)
        
        for i in range(HIDDENNEURONS):
            pygame.draw.circle(GD, black, (NETWORKPOSITION[0] + LAYERSPACING, NETWORKPOSITION[1] + ((i+1)*NEURONSPACING)), NEURONSIZE)
            pygame.draw.circle(GD, white, (NETWORKPOSITION[0] + LAYERSPACING, NETWORKPOSITION[1] + ((i+1)*NEURONSPACING)), NEURONSIZE-NEURONOUTLINE)

        for i in range(OUTPUTNEURONS):
            pygame.draw.circle(GD, black, (NETWORKPOSITION[0] + LAYERSPACING * 2, NETWORKPOSITION[1] + ((i+1)*NEURONSPACING)), NEURONSIZE)
            if i in self.KEYS:
                pygame.draw.circle(GD, red, (NETWORKPOSITION[0] + LAYERSPACING * 2, NETWORKPOSITION[1] + ((i+1)*NEURONSPACING)), NEURONSIZE-NEURONOUTLINE)
            else:
                pygame.draw.circle(GD, white, (NETWORKPOSITION[0] + LAYERSPACING * 2, NETWORKPOSITION[1] + ((i+1)*NEURONSPACING)), NEURONSIZE-NEURONOUTLINE)

# Hier wordt enkel de eerste keer een groep auto's aangemaakt en hun neural networks aangemaakt.
spawnGen()
for i in range(len(currentGen)):
    currentGen[i].setupNetwork()

# Dit is de Update() loop die elke frame gerunned wordt en alle logica geroepen wordt.
running = True
while running:
    # Hier maken we de achtergrond wit.
    GD.fill(white)
    # Dit fungeert als backup voor het sluiten van het programma.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Hier wordt elke auto bewogen en getekend mits het nog leeft.
    for i in range(len(currentGen)):
        if not currentGen[i].dead:
            currentGen[i].move()
            currentGen[i].draw()
    
    # Hier wordt de finishlijn, de baan en checkpoints over de auto heen getekend.
    GD.blit(finishline, (450, 435))
    GD.blit(track, (0, 0))
    GD.blit(checkpoint, (0, 0))

    # Hier wordt gekeken wanneer een auto een finishlijn of checkpoint voorbij gaat mits hij niet dood is.
    for i in range(len(currentGen)):
        if not currentGen[i].dead:
            currentGen[i].checkRewardCollision()


    allDead = False
    cooldownid += 1
    # De cooldown zorgt ervoor dat we de zwaarste berekeningen niet elke frame uitvoeren, maar elke 6 frames. Met 60 fps is dat 10x per seconde.
    if cooldownid >= cooldown:
        cooldownid = 0
        allDead = True
        # Hier kijken we voor elke auto of ze gebotst zijn met een muur, hoeveel pixels er zijn tussen de muren (input) en voeren we het neural network uit mits hij niet dood is.
        for i in range(len(currentGen)):
            if not currentGen[i].dead:
                currentGen[i].checkCollision()
                currentGen[i].getInput()
                currentGen[i].runNetwork()
                allDead = False
    # Als er een auto nog leeft dan wordt allDead false en wordt deze functie niet uitgevoerd, als alle autos dood zijn wordt reset() uitgevoerd.
    if allDead:
        reset()

    # Nu de collision is gecheckt voor de zwarte baan (hij checkt alleen voor een zwarte pixel) kunnen we de mooie baan tekenen.
    GD.blit(greentrack, (0, 0))

    # We checken voor elke levende auto wie de hoogste fitness heeft en tekenen vervolgens dit neural network.
    highestFitness = -1
    highestFitnessID = -1
    for i in range(len(currentGen)):
        if not currentGen[i].dead:
            if currentGen[i].fitness > highestFitness:
                highestFitness = currentGen[i].fitness
                highestFitnessID = i

    # Als er geen beste neural network is tekenen we er geen
    if not highestFitnessID == -1:
        currentGen[highestFitnessID].drawNetwork()

    # Dit update het scherm en zorgt ervoor dat we 60 fps krijgen.
    pygame.display.update()
    pygame.display.flip()
    clock.tick(60)

# Als het programma afsluit printen we nog een keer de beste fitness van deze generatie, de beste fitness ooit en de fitnesses van de huidige fitnesses van alle levende autos.
print("Best fitness: " + str(bestFitness))
print("Elite fitness: " + str(eliteFitness))
for i in range(len(currentGen)):
    if not currentGen[i].dead:
        print("Fitnesses: " + str(currentGen[i].fitness))