# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from game import Actions
from util import nearestPoint
from pacman import GameState
#################
# Team creation #
#################
beliefs = {}
Target = None
lastYuanliPos = None
chased = False
visitedPos = set()
def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########
class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
      
  def registerInitialState(self, gameState):

    CaptureAgent.registerInitialState(self, gameState)
    self.CurrentState = gameState
    self.start = gameState.getAgentPosition(self.index)
    if(self.index == 0 or self.index == 2):
      self.red = True
    self.allyIndex = self.getAllyIndex(gameState)
    food = self.getFood(gameState)
    self.upperFood = []
    self.lowerFood = []
    self.wall = gameState.getWalls().asList()
    self.x,self.y = self.wall[len(self.wall)-1]
    self.splitFood(food)
    self.legalPositions = [pos for pos in gameState.getWalls().asList(False)]
    self.midLane = self.calculateMidLane(self.x, self.y)
    self.count = 0
    self.foodRest = []
    self.invaders = util.Counter()
    self.enemyPos = ParticleFilter(self, gameState)
    self.freedomZone = False
    
  def getAllyIndex(self, gameState):

    enemy = self.getOpponents(gameState)
    players = [0,1,2,3]
    allies = [x for x in players if x not in enemy]
    allies.remove(self.index)
    return allies[0]
    
  def splitFood(self, food):

    mid = self.y // 2
    self.lowerFood = []
    self.upperFood = []
    for fo in food.asList():
      if(fo[1] < mid):
        self.lowerFood.append(fo)
      else:
        self.upperFood.append(fo)

  def calculateMidLane(self, x, y):

    midLaneCoord = []
    for t in range(y):
      if(self.red):
        if(((x//2),t)not in self.wall):
          midLaneCoord.append(((x//2),t))
      else:
        if(((x//2)+1,t) not in self.wall):
          midLaneCoord.append(((x//2)+1,t))
    return midLaneCoord

  def chooseAction(self, gameState):
    
    global Target
    # print self.index, Target
    self.enemiesIndex = self.getOpponents(gameState)
    self.splitFood(self.getFood(gameState))
    self.enemyPos.observe(self, gameState)
    actions = gameState.getLegalActions(self.index)
    foodLeft = len(self.getFood(gameState).asList())
    self.prevObserve = self.getPreviousObservation()
    if self.prevObserve:
      self.prevPos = self.prevObserve.getAgentPosition(self.index)
    else:
      self.prevPos = gameState.getAgentPosition(self.index)
    # Back home
    if foodLeft <= 2:
      bestDist = 999999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist

      return bestAction
    
    self.invaders = {}
    self.enemyPosDict = {}
    for enemy in self.getOpponents(gameState):
      enemPs = self.enemyPos.getPos(self, gameState, enemy)
      self.enemyPosDict[enemy] = enemPs
      if(gameState.getAgentState(enemy).isPacman):
        self.invaders[enemy] = enemPs
    self.enemyPos.elapseTime()
    #determine attack or defense
    if(gameState.getAgentState(self.index).scaredTimer == 0) :
      if(len(self.invaders)>=1):
        return self.getDefenseAction(self.invaders, gameState)
      elif(len(self.invaders)==0):
        Target = None
    actions = gameState.getLegalActions(self.index)
    actions.remove(Directions.STOP)
    maxQ = -999999999
    appliedAction = None
 
    for action in actions:
      successor = self.getSuccessor(gameState, action)
      myNextPos = successor.getAgentState(self.index).getPosition()
      result = self.evaluate(gameState, action)
      if result > maxQ:
        maxQ = result
        appliedAction = action
    
    successor = self.getSuccessor(gameState, appliedAction) if appliedAction else self.getSuccessor(gameState, Directions.STOP)
    myNextPos = successor.getAgentState(self.index).getPosition()

    if(myNextPos in self.getFood(gameState).asList()):
        self.count+=1

    return appliedAction

  def getDefenseAction(self, enemyPos, gameState):
    self.updateFreedomZone(gameState)
    if self.freedomZone:
      action = self.DefenseAction(enemyPos, gameState)
    else:
      action = self.goToFreedomZone(gameState, enemyPos)
    return action

  def DefenseAction(self, enemiesPos, gameState):
    
    from util import Queue
    from game import Directions
    global Target

    # print 'start dfense: ', self.index

    Target = None
    distances = util.Counter()
    fromWhere = {}
    pathList = []
    
    ghostEnemyPos = []
    for enemy in self.getOpponents(gameState):
      if not gameState.getAgentState(enemy).isPacman:
        ghostEnemyPos.append(self.enemyPosDict[enemy])

    if not Target:
        # print 'no taget'
        for enemy, enemyPos in enemiesPos.iteritems():
            distances[enemy] = self.getMazeDistance(gameState.getAgentPosition(self.index), enemyPos)
            # print self.index, 'searching target'
        minEnemyDistance = 99999999999
        for index, distance in distances.iteritems():
          if distance < minEnemyDistance:
            minEnemyDistance = distance
            Target = index
        # print 'agent:', self.index, 'target found:', Target

    # global visitedPos
    stateQueue = Queue()
    visitedState = set()
    # print gameState
    startState = (gameState.getAgentPosition(self.index), enemiesPos[Target])
    #print gameState.getAgentPosition(self.index), enemiesPos[Target]
    stateQueue.push(startState)
    visitedState.add(startState)
    fromWhere[startState] = None

    while not stateQueue.isEmpty():
        #print 'finding path'
        currentState = stateQueue.pop()
        
        if currentState[0] == currentState[1]:
            #print 'find path to target'
            while fromWhere[currentState] is not None:
              # print 'generate action'
              pathList.append(fromWhere[currentState][1])
              currentState = fromWhere[currentState][0]
            # print 'action generated' 
            # print 'pathlist length: ', len(pathList) 
            pathList.reverse()
            # print 'find action to target:', pathList
            if pathList:
              # print pathList[0]
              return pathList[0]
            else:
              # print Directions.STOP
              return Directions.STOP

        tempLegalActions = []
        oldPos = currentState[0]
        possibleNewPositions = [(oldPos[0]+i, oldPos[1]+j) for i in [-1,0,1] for j in [-1,0,1] if not (abs(i) == 1 and abs(j) == 1)]
        
        for p in possibleNewPositions:
          if p in self.legalPositions:
            tempLegalActions.append(p)

        for a in tempLegalActions:
            # print a
            act = self.positionToDirection(oldPos, a)
            # print a
            # sucState = currentState.generateSuccessor(self.index, act)
            sucState = (a, enemiesPos[Target])
            # nextPos = sucState.getAgentPosition(self.index)
            if len(tempLegalActions) > 2:
              # print 'index: ', self.index, 'have', len(tempLegalActions), 'validChoices'
              if sucState not in visitedState and sucState[0] not in ghostEnemyPos and sucState[0] != gameState.getAgentPosition(self.getAllyIndex(gameState)):
                  # print 'new succ pushed'
                  stateQueue.push(sucState)
                  visitedState.add(sucState)
                  fromWhere[sucState] = (currentState, act)
            else:
              # print 'index: ', self.index, 'following ally'
              if sucState not in visitedState and sucState[0] not in ghostEnemyPos:
                  # print 'new succ pushed'
                  stateQueue.push(sucState)
                  visitedState.add(sucState)
                  fromWhere[sucState] = (currentState, act)
            
            
    
    tempLegalActions = []
    oldPos = gameState.getAgentPosition(self.index)
    possibleNewPositions = [(oldPos[0]+i, oldPos[1]+j) for i in [-1,0,1] for j in [-1,0,1] if not (abs(i) == 1 and abs(j) == 1)]
        
    for p in possibleNewPositions:
      if p in self.legalPositions:
        tempLegalActions.append(p)
    for a in tempLegalActions:
      act = self.positionToDirection(oldPos, a)

    return act

  def positionToDirection(self, oldPos, newPos):

    if newPos[0] - oldPos[0] > 0:
      return Directions.EAST
    elif newPos[0] - oldPos[0] < 0:
      return Directions.WEST
    elif newPos[1] - oldPos[1] > 0:
      return Directions.NORTH
    elif newPos[1] - oldPos[1] < 0:
      return Directions.SOUTH
    elif oldPos == newPos:
      return Directions.STOP

  def getSuccessor(self, gameState, action):

    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()

    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)

    else:

      return successor

  def evaluate(self, gameState, action):
    
    """
    Computes a linear combination of features and feature weights
    """

    
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    #if(self.index == 3):
      #print gameState.getAgentPosition(self.index)
      #print features
      #print weights
     
    return features * weights

  def getFeatures(self, gameState, action):

    # print 'start attack: ', self.index
    features = util.Counter()
    
    successor = self.getSuccessor(gameState, action)
    myNextPos = successor.getAgentState(self.index).getPosition()

    nextFoodList = self.getFood(successor).asList()
    foodList = self.getFood(gameState).asList()  

    capsuleList = self.getCapsules(gameState)
    nextCapsuleList = self.getCapsules(successor)

    if capsuleList:
      nextStepDistance2NearestCapsule = min([self.getMazeDistance(myNextPos, capsule) for capsule in capsuleList])
    
    enemy = self.getOpponents(gameState)
    CurrentPos = gameState.getAgentPosition(self.index)
   
    features['successorScore'] = -len(nextFoodList)
    visiableEnemies = [gameState.getAgentPosition(i) for i in self.enemiesIndex]
    enemyAround = False
    
    for i in range(len(visiableEnemies)):

        if visiableEnemies[i] != None:

            if self.getMazeDistance(CurrentPos, visiableEnemies[i]) < 4 and gameState.getAgentState(self.enemiesIndex[i]).scaredTimer<3:
                
                enemyAround = True

    if not enemyAround or self.count < 6 :
      
      if len(foodList) > 0: # This should always be True,  but better safe than sorry. AND I AM FART
        if(self.index == 0 or self.index ==1):
          if(len(self.upperFood)!=0):
            minDistance = min([self.getMazeDistance(myNextPos, food) for food in self.upperFood])
            features['distanceToFood'] = minDistance
        #   elif self.beChased(gameState):
        #     if capsuleList:
        #       features['distanceToFood'] = -1 * nextStepDistance2NearestCapsule
          else:
            self.upperFood = self.lowerFood
        else:
          if(len(self.lowerFood)!=0):
            minDistance = min([self.getMazeDistance(myNextPos, food) for food in self.lowerFood])
            features['distanceToFood'] = minDistance
        #   elif self.beChased(gameState):
        #     if capsuleList:
        #       features['distanceToFood'] = -1 * nextStepDistance2NearestCapsule
          else:
            self.lowerFood = self.upperFood
     
      minDistance2Enemy = 99999

      for pos in self.enemyPosDict.values():
        tempDist = self.getMazeDistance(myNextPos, pos)
        if tempDist < minDistance2Enemy:
          minDistance2Enemy = tempDist

      features['enemyMinDistance'] = tempDist

      for co in range(len(enemy)):
        if (gameState.getAgentPosition(enemy[co])!=None):
          distanceToEnemy = self.getMazeDistance(myNextPos, gameState.getAgentPosition(enemy[co]))
          if distanceToEnemy <= 4 and gameState.getAgentState(enemy[co]).isPacman == False and gameState.getAgentState(enemy[co]).scaredTimer < distanceToEnemy / 2:
            features['enemyAround'] = distanceToEnemy
            features['validChoices'] = len(successor.getLegalActions(self.index))
            features['dangerPlace'] = features['enemyAround'] + features['validChoices']
      
      features['leftFoods'] = len(nextFoodList)

      if self.beChased(gameState):
        print 'beChased confirmed!!'
        if capsuleList:
              print 'setting capsule'
              features['distance2Capsule'] = nextStepDistance2NearestCapsule
              features['nextLeftCapsule'] = len(nextCapsuleList)
              features['dangerPlace'] = 0
    #   print 'Max feature: ', features.argMax()
    #   distanceToAlly = self.getMazeDistance(myNextPos, self.getAllyPosition(gameState))
 
    #   if gameState.getAgentState(self.index).isPacman:
    #     features['distanceToAlly'] = distanceToAlly + 0.2
    #   else:
    #     features['distanceToAlly'] = 0

      # print "index: ", self.index, "features[distanceToFood]: ", features['distanceToFood'], "features[distanceToMidLane]: ", features['distanceToMidLane'], "features[leftFoods]: ", features['leftFoods'], "features['dangerPlace']: ",features['dangerPlace'], "features['enemyMinDistance']: ", features['enemyMinDistance']
    else: #if pacman eat more than 6 dots, then he needs go back
      # print 'index: ', self.index, 'back home'
      minDistance = min([self.getMazeDistance(myNextPos, mid) for mid in self.midLane])

      features['distanceToMidLane'] = minDistance
      
      minDistance2Enemy = 99999
      for pos in self.enemyPosDict.values():
        tempDist = self.getMazeDistance(myNextPos, pos)
        if tempDist < minDistance2Enemy:
          minDistance2Enemy = tempDist
          features['enemyMinDistance'] = tempDist

      for co in range(len(enemy)):
        if (gameState.getAgentPosition(enemy[co])!=None):
          distanceToEnemy = self.getMazeDistance(myNextPos, gameState.getAgentPosition(enemy[co]))
          if distanceToEnemy <= 4 and gameState.getAgentState(enemy[co]).isPacman == False and gameState.getAgentState(enemy[co]).scaredTimer < distanceToEnemy / 2:
            features['enemyAround'] = distanceToEnemy
            # features['validChoices'] = len(successor.getLegalActions(self.index))
            # features['dangerPlace'] = features['enemyAround'] * features['validChoices']
      if(self.onOurside(CurrentPos)):

        self.count = 0

    return features

  def onOurside(self, position):
    if(self.red):
      if position[0] <= self.midLane[0][0]:
        return True
      else:
        return False
    else:
      if (position[0] >= self.midLane[0][0]):
        return True
      else:
        return False
  def getWeights(self, gameState, action):
    # return {'successorScore': 10, 'eatFood': 10, 'distanceToFood': -1, 
    # 'Wantcapsules': 20, 'distanceToMidLane': -1, 'distanceToAlly': 2}, 'leftFoods': -1, 'validChoices': 1,'enemyMinDistance': 5,
    return {'distanceToFood':-2, 'distanceToMidLane': -1,  'leftFoods': -1, 'dangerPlace': 50, 'enemyMinDistance': 1, 'distance2Capsule': -2, 'nextLeftCapsule': -5, 'enemyAround': 10}
  
  def beChased(self, gameState):
    
    global chased

    prevState = self.getPreviousObservation()

    if prevState:
        prevPos = prevState.getAgentPosition(self.index)
    else:
        chased = False
        return False

    currentPos = gameState.getAgentPosition(self.index)

    for enemy in self.getOpponents(gameState):

      prevEnemyPos = prevState.getAgentPosition(enemy)
      currentEnemyState = not gameState.getAgentState(enemy).isPacman and gameState.getAgentState(enemy).scaredTimer < 3
      currentEnemyPos = gameState.getAgentPosition(enemy)
      if prevEnemyPos and currentEnemyPos:
        print prevEnemyPos
        print currentEnemyPos
        print self.getMazeDistance(prevPos, prevEnemyPos)
        print self.getMazeDistance(currentPos, currentEnemyPos)
        if self.getMazeDistance(prevPos, prevEnemyPos) >= self.getMazeDistance(currentPos, currentEnemyPos) and currentEnemyState:
            print 'be chased'
            chased = True
            break

        else:

            chased = False
      else:

        chased = False

    return chased

  def updateFreedomZone(self, gameState):
    
    index = self.index
    tempLegalActions = []
    oldPos = gameState.getAgentPosition(index)
    possibleNewPositions = [(oldPos[0]+i, oldPos[1]+j) for i in [-1, 0, 1] for j in [-1,0,1] if not (abs(i) == 1 and abs(j) == 1)]
    
    for p in possibleNewPositions:

      if p in self.legalPositions:

        tempLegalActions.append(p)
    
    if gameState.getAgentPosition(index) == gameState.getInitialAgentPosition(index) and len(tempLegalActions) < 4:

      self.freedomZone = False

    elif len(tempLegalActions) > 3:
      
      self.freedomZone = True
      
  def goToFreedomZone(self, gameState, enemiesPos):

    from util import Queue
    from game import Directions
    # global Target
    destinition = gameState.getInitialAgentPosition(self.enemiesIndex[0])
    # print 'start dfense: ', self.index
    # Target = None
    fromWhere = {}
    pathList = []
    # ghostEnemyPos = []
    # for enemy in self.getOpponents(gameState):
    #   if not gameState.getAgentState(enemy).isPacman:
    #     ghostEnemyPos.append(self.enemyPosDict[enemy])

    # if not Target:
    #     # print 'no taget'
    #     for enemy, enemyPos in enemiesPos.iteritems():
    #         distances[enemy] = self.getMazeDistance(gameState.getAgentPosition(self.index), enemyPos)
    #         # print self.index, 'searching target'
    #     minEnemyDistance = 99999999999
    #     for index, distance in distances.iteritems():
    #       if distance < minEnemyDistance:
    #         minEnemyDistance = distance
    #         Target = index
    #     # print 'agent:', self.index, 'target found:', Target

    # global visitedPos
    stateQueue = Queue()
    visitedState = set()
    # print gameState
    startState = gameState.getAgentPosition(self.index)
    # print gameState.getAgentPosition(self.index), enemiesPos[Target]
    stateQueue.push(startState)
    visitedState.add(startState)
    fromWhere[startState] = None

    while not stateQueue.isEmpty():
        # print 'finding path'
        currentState = stateQueue.pop()
        
        if currentState == destinition:
            # print 'find path to target'
            while fromWhere[currentState] is not None:
              # print 'generate action'
              pathList.append(fromWhere[currentState][1])
              currentState = fromWhere[currentState][0]
            # print 'action generated' 
            # print 'pathlist length: ', len(pathList) 
            pathList.reverse()
            # print 'find action to target:', pathList
            # if pathList:
              # print pathList[0]
            return pathList[0]

            # else:
            #   # print Directions.STOP
            #   return Directions.STOP

        tempLegalActions = []
        oldPos = currentState
        possibleNewPositions = [(oldPos[0]+i, oldPos[1]+j) for i in [-1, 0, 1] for j in [-1,0,1] if not (abs(i) == 1 and abs(j) == 1)]
        
        for p in possibleNewPositions:
          if p in self.legalPositions:
            tempLegalActions.append(p)

        for a in tempLegalActions:
            # print a
            act = self.positionToDirection(oldPos, a)
            # print a
            # sucState = currentState.generateSuccessor(self.index, act)
            sucState = a
            # nextPos = sucState.getAgentPosition(self.index)
           
            if sucState not in visitedState:
                # print 'new succ pushed'
                stateQueue.push(sucState)
                visitedState.add(sucState)
                fromWhere[sucState] = (currentState, act)
            # else:
            #   if sucState not in visitedState and sucState[0] not in ghostEnemyPos:
            #       # print 'new succ pushed'
            #       stateQueue.push(sucState)
            #       visitedState.add(sucState)
            #       fromWhere[sucState] = (currentState, act)
    
    tempLegalActions = []
    oldPos = gameState.getAgentPosition(self.index)
    possibleNewPositions = [(oldPos[0]+i, oldPos[1]+j) for i in [-1,0,1] for j in [-1,0,1] if not (abs(i) == 1 and abs(j) == 1)]
        
    for p in possibleNewPositions:
      if p in self.legalPositions:
        tempLegalActions.append(p)
    for a in tempLegalActions:
      act = self.positionToDirection(oldPos, a)

    return act
# class Attack(DummyAgent):
#   def Attacker():
#     print "im attacker"

# class Defense(DummyAgent):
#   def defense():
#     return None

class ParticleFilter(object):
    
  def __init__(self, agent, gameState):

    global beliefs
    self.agent = agent
    self.gameState = gameState
    self.midWidth = gameState.data.layout.width/2.0
    # self.index = self.agent.index
    self.legalPositions = [pos for pos in self.gameState.getWalls().asList(False)]
    self.enemies = self.agent.getOpponents(self.gameState)
    beliefs = {}
    for enemy in self.enemies:
      beliefs[enemy] = util.Counter()
    #   for pos in self.legalPositions:
    #     beliefs[enemy][pos] = 1.0
    for enemy in self.enemies:
      beliefs[enemy][self.gameState.getInitialAgentPosition(enemy)] = 1.0
      beliefs[enemy].normalize()

  def elapseTime(self):
    
    global beliefs
    global Target
    tempDist = {}

    for enemy, oldPosDist in beliefs.iteritems():
      tempDist[enemy] = util.Counter()

      for oldPos in oldPosDist.keys():
        nextMove = util.Counter()
        possibleNewPositions = [(oldPos[0]+i, oldPos[1]+j) for i in [-1,0,1] for j in [-1,0,1] if not (abs(i) == 1 and abs(j) == 1)]
        
        for p in possibleNewPositions:
          if p in self.legalPositions:
            nextMove[p] = 1.0
        
        nextMove.normalize()
        
        for p, prob in nextMove.iteritems():
          tempDist[enemy][p] += prob * beliefs[enemy][oldPos]
      
      tempDist[enemy].normalize()
    
    beliefs = tempDist

  def observe(self, agent, gameState):
    
    global beliefs
    global visitedPos
    global Target
    newDist = {}
    myPos = gameState.getAgentPosition(agent.index)
    for enemy in self.enemies:

      newDist[enemy] = util.Counter()
      
      if gameState.getAgentPosition(enemy):

        newDist[enemy][gameState.getAgentPosition(enemy)] = 1.0
        continue
      
      prevState = agent.getPreviousObservation()
      
      if prevState:
        prevFood = set(agent.getFoodYouAreDefending(prevState).asList(True))
        currentFood = set(agent.getFoodYouAreDefending(gameState).asList(True))
        eatenFood = prevFood - currentFood
        if eatenFood:
          if gameState.getAgentState(enemy).isPacman:
            for f in eatenFood:
              newDist[enemy][f] = 1.0
            continue
    
      if prevState and prevState.getAgentPosition(enemy) and gameState.getAgentPosition(enemy) == None and util.manhattanDistance(myPos,prevState.getAgentPosition(enemy)) < 3:

          newDist[enemy][gameState.getInitialAgentPosition(enemy)] = 1.0
          Target = None
          visitedPos = set()

          continue

      else:

        for p in self.legalPositions:

          trueDistance = util.manhattanDistance(myPos, p)
          noisyDistance = gameState.getAgentDistances()
          distanceProb = gameState.getDistanceProb(trueDistance, noisyDistance[enemy])
          newDist[enemy][p] = beliefs[enemy][p] * distanceProb

          if agent.red:
            
            ispac = p[0] < self.midWidth
          
          else:
            
            ispac = p[0] > self.midWidth
          
          if ispac != gameState.getAgentState(enemy).isPacman:
            
            newDist[enemy][p] = 0

          if trueDistance <= 5:

            newDist[enemy][p] = 0

              
        newDist[enemy].normalize()
    
    beliefs = newDist
    #self.agent.displayDistributionsOverPositions(beliefs.values())

  def getPos(self, agent, gameState, enemy):
    
    bestPos = beliefs[enemy].argMax()
    #color = [0,0,0]
    #if max(self.enemies) > 3:
    #  for enemy in self.enemies:
    #
    #    color[enemy - 2] = 1
    #else:
    #  color[enemy - 1] = 1
    #agent.debugDraw(bestPos, color, clear=True)
    # print 'enemy', bestPos
    return bestPos

