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
import capture
import random, time, util
from game import Directions
import game
from util import nearestPoint
import math
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
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


# tunnels will store all tunnel positions of the map, the tunnel means 
#only one way to leave
tunnels = []   

# tunnels will store all tunnel positions of the map, but regarding the 
# boundary as wall to find tunnels. 
defensiveTunnels = []

# store the walls of the map
walls = []

"""
getAllTunnels will return the all tunnels as a list, it uses a while loop 
#to find a tunnel level by level, stop until no more tunnels in the map
"""
def getAllTunnels(legalPositions):
    tunnels = []
    while len(tunnels) != len(getMoreTunnels(legalPositions, tunnels)):
        tunnels = getMoreTunnels(legalPositions, tunnels)
    return tunnels

"""
getMoreTunnels is the function to find the next level's tunnel
"""
def getMoreTunnels(legalPositions, tunnels):
    newTunnels = tunnels
    for i in legalPositions:
        neighborTunnelsNum = getSuccsorsNum(i, tunnels)
        succsorsNum = getSuccsorsNum(i, legalPositions)
        if succsorsNum - neighborTunnelsNum == 1 and i not in tunnels:
            newTunnels.append(i)
    return newTunnels

"""
getMoreTunnels is the function to find the next level's tunnel
"""
def getSuccsorsNum(pos, legalPositions):
    num = 0
    x, y = pos
    if (x + 1, y) in legalPositions:
        num += 1
    if (x - 1, y) in legalPositions:
        num += 1
    if (x, y + 1) in legalPositions:
        num += 1
    if (x, y - 1) in legalPositions:
        num += 1
    return num

"""
getSuccsorsPos will return all position's legal neighbor positions
"""
def getSuccsorsPos(pos, legalPositions):
    succsorsPos = []
    x, y = pos
    if (x + 1, y) in legalPositions:
        succsorsPos.append((x + 1, y))
    if (x - 1, y) in legalPositions:
        succsorsPos.append((x - 1, y))
    if (x, y + 1) in legalPositions:
        succsorsPos.append((x, y + 1))
    if (x, y - 1) in legalPositions:
        succsorsPos.append((x, y - 1))
    return succsorsPos

"""
given current position and an action, nextPos will return the next position
"""
def nextPos(pos, action):
  x, y = pos
  if action == Directions.NORTH:
    return (x, y + 1)
  if action == Directions.SOUTH:
    return (x, y - 1)
  if action == Directions.EAST:
    return (x + 1, y)
  if action == Directions.WEST:
    return (x - 1, y)
  return pos

"""
manhattanDist: input two points, return the mahattan distance between 
these two points
"""
def manhattanDist(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x2 - x1) + abs(y2 - y1)

"""
getTunnelEntry: given a position, if position in tunnels, it will return
the entry position of this tunnel
"""
def getTunnelEntry(pos, tunnels, legalPositions):
    if pos not in tunnels:
        return None
    aTunnel = getATunnel(pos, tunnels)
    for i in aTunnel:
        possibleEntry = getPossibleEntry(i, tunnels, legalPositions)
        if possibleEntry != None:
            return possibleEntry

"""
getPossibleEntry: this assisted funtion used in getTunnelEntry to
find if next neighbor position is tunnel entry
"""
def getPossibleEntry(pos, tunnels, legalPositions):
    x, y = pos
    if (x + 1, y) in legalPositions and (x + 1, y) not in tunnels:
        return (x + 1, y)
    if (x - 1, y) in legalPositions and (x - 1, y) not in tunnels:
        return (x - 1, y)
    if (x, y + 1) in legalPositions and (x, y + 1) not in tunnels:
        return (x, y + 1)
    if (x, y - 1) in legalPositions and (x, y - 1) not in tunnels:
        return (x, y - 1)
    return None

"""
getATunnel: input a position and tunnels, this function will return a tunnel
that this position belongs to
"""
def getATunnel(pos, tunnels):

    if pos not in tunnels:
        return None
    bfs_queue = util.Queue()
    closed = []
    bfs_queue.push(pos)
    while not bfs_queue.isEmpty():
        currPos = bfs_queue.pop()
        if currPos not in closed:
            closed.append(currPos)
            succssorsPos = getSuccsorsPos(currPos, tunnels)
            for i in succssorsPos:
                if i not in closed:
                    bfs_queue.push(i)
    return closed


"""
class node: used in UCTs process. Nodes are different gameStates, it has
some functions:
addChild, to add a child node. 
findParnt: find this node's parent node. 
chooseChild: choose the child with highest UCT value
"""
class Node:
    def __init__(self, value, id=0):
        (gameState, t, n) = value
        self.id = id
        self.children = []
        self.value = (gameState, float(t), float(n))
        self.isLeaf = True

    def addChild(self, child):
        self.children.append(child)

    def chooseChild(self):
        _, _, pn = self.value
        maxUCB = -999999
        bestChild = None
        for i in self.children:
            _, t, n = i.value
            if n == 0:
                return i
            UCB = t + 1.96 * math.sqrt(math.log(pn) / n)
            if maxUCB < UCB:
                maxUCB = UCB
                bestChild = i
        return bestChild

    def findParent(self, node):
        for i in self.children:
            if i == node:
                return self
            else:
                possibleParent = i.findParent(node)
                if possibleParent != None:
                    return possibleParent

    def __str__(self):
        (_, t, n) = self.value
        id = self.id
        return "Node " + str(id) + ", t = " + str(t) + ", n = " + str(n)


"""
class tree: used in UCTs process, to store nodes, it has some functions:
insert: given a parent node and child node, add this child node to the tree
getParent: uses findParent function to return the parent node
backPropagate: standard UCT backpropagate process, do update job
select: iteratively find the child with largest UCT value

"""
class Tree:
    def __init__(self, root):
        self.count = 1
        self.tree = root
        self.leaf = [root.value[0]]

    def insert(self, parent, child):
        id = self.count
        self.count += 1
        child.id = id
        parent.addChild(child)
        if parent.value[0] in self.leaf:
            self.leaf.remove(parent.value[0])
        parent.isLeaf = False
        self.leaf.append(child.value[0])

    def getParent(self, node):
        if node == self.tree:
            return None
        return self.tree.findParent(node)

    def backPropagate(self, r, node):
        (gameState, t, n) = node.value
        node.value = (gameState, t + r, n + 1)
        parent = self.getParent(node)
        if parent != None:
            self.backPropagate(r, parent)

    def select(self, node = None):
        if node == None:
            node = self.tree
        if not node.isLeaf:
            nextNode = node.chooseChild()
            return self.select(nextNode)
        else:
            return node



    """
    This class generates beliefs of positions of invaders 
    HMM model is applied in this class

    """
class ParticleFilter:
    
  def __init__(self, agent, gameState):
    
    self.start = gameState.getInitialAgentPosition(agent.index)
    self.agent = agent
    self.midWidth = gameState.data.layout.width/2
    self.legalPositions = [p for p in gameState.getWalls().asList(False)]
    self.enemies = self.agent.getOpponents(gameState)
    self.beliefs = {}
    for enemy in self.enemies:
        self.beliefs[enemy] = util.Counter()
        self.beliefs[enemy][gameState.getInitialAgentPosition(enemy)] = 1.0
        self.beliefs[enemy].normalize()


  
  # This function updates the distribution of invaders'
  # location with uniformed distribution
  def elapseTime(self):

    for enemy in self.enemies:
        dist = util.Counter()

        for p in self.legalPositions:
            newDist = util.Counter()

            allPositions = [(p[0]+i, p[1]+j) for i in [-1,0,1] for j in [-1,0,1] if not (abs(i) == 1 and abs(j) == 1)]

            for q in self.legalPositions:
                if q in allPositions:
                    newDist[q] = 1.0
            newDist.normalize()

            for pos, probability in newDist.items():
                dist[pos] = dist[pos] + self.beliefs[self.enemy][pos] * probability

        dist.normalize()
        self.beliefs[enemy] = dist

  # This function uses noisy distance to narrow the range
  # of probabilities of invaders' location
  def observe(self, agent, gameState):

      myPos = gameState.getAgentPosition(agent.index)
      noisyDistance = gameState.getAgentDistances()

      dist = util.Counter()

      for enemy in self.enemies:
          for pos in self.legalPositions:

              trueDistance = util.manhattanDistance(myPos, pos)
              probability = gameState.getDistanceProb(trueDistance, noisyDistance)

              if agent.red:
                  ifPacman = pos[0] < self.midWidth
              else:
                  ifPacman = pos[0] > self.midWidth

              if trueDistance <= 6 or ifPacman != gameState.getAgentState(enemy).isPacman:
                  dist[pos] = 0.0
              else:
                  dist[pos] = self.beliefs[enemy][pos] * probability

          dist.normalize()
          self.beliefs[enemy] = dist

  def getPossiblePosition(self, enemy):
    
      pos = self.beliefs[enemy].argMax()
      return pos

class ReflexCaptureAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.

    '''

    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

    self.changeEntrance = False                                         # control the change entrance feature
    self.nextEntrance = None                                            # if need change entrance, it stores the next entrace's position
    self.carriedDot = 0                                                 # store the dots the offensive agent carried
    self.tunnelEntry = None                                             # if agents in tunnel, store this tunnel entry 
    global walls                                                        # declare the global type
    global tunnels                                                      # declare the global type
    global openRoad                                                     # declare the global type, openRoad are the places that are not in tunnels
    global legalPositions                                               # declare the global type
    walls = gameState.getWalls().asList()
    if len(tunnels) == 0:
      legalPositions = [p for p in gameState.getWalls().asList(False)]
      tunnels = getAllTunnels(legalPositions)
      openRoad = list(set(legalPositions).difference(set(tunnels)))
    self.capsule = None                                                 # store the safe capsule that agent will run to
    self.nextOpenFood = None                                            # store the nearest safe food in open road that agent will run to
    self.nextTunnelFood = None                                          # store the nearest safe food in tunnel that agent will run to
    self.runToBoundary = None                                           # store the nearest boundary position
    self.stuckStep = 0                                                  # count the step if our agent is stuck with opponents
    self.curLostFood = None                                             # store the current food that eaten by invader 
    self.ifStuck = False                                                # When stuck found, this change to true, start to count steps
    self.enemyGuess = ParticleFilter(self, gameState)                   # Store the invaders guessed position
    self.invadersGuess = False                                          # if found invaders, this will change to True
    global defensiveTunnels                                             # declare the global type
    width = gameState.data.layout.width
    legalRed = [p for p in legalPositions if p[0] < width / 2]          # legal positions in red area
    legalBlue = [p for p in legalPositions if p[0] >= width / 2]        # legal positions in blue area

    if len(defensiveTunnels) == 0:
        if self.red:
            defensiveTunnels = getAllTunnels(legalRed)
        else:
            defensiveTunnels = getAllTunnels(legalBlue)


    """
    chooseAction: if self.ifStuck is True, it will call MCTs function to make decision.
    otherwise it will find the best action using evaluate function
    """

  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)

    values = [self.evaluate(gameState, a) for a in actions]


    Q = max(values)

    if self.ifStuck:
        return self.simulation(gameState)

    bestActions = [a for a, v in zip(actions, values) if v == Q]

    action = random.choice(bestActions)

    return action


    """
    Finds the next successor which is a grid position (location tuple).
    """
  def getSuccessor(self, gameState, action):

    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

    """
    Computes a linear combination of features and feature weights
    """
  def evaluate(self, gameState, action):

    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)

    return features * weights
    

    """
    if agent in tunnel entry, this method will be called to evaluate this
    tunnel. if no food in this tunnel, return 0. Otherwise return the
    distance between nearest food in tunnel with agent
    """
  def ifWasteTunnel(self, gameState, successor):

    curPos = gameState.getAgentState(self.index).getPosition()
    sucPos = successor.getAgentState(self.index).getPosition()
    if curPos not in tunnels and sucPos in tunnels:

      self.tunnelEntry = curPos

      dfs_stack = util.Stack()
      closed = []
      dfs_stack.push((sucPos, 1))

      while not dfs_stack.isEmpty():
        (x, y), length = dfs_stack.pop()
        if self.getFood(gameState)[int(x)][int(y)]:
          return length

        if (x, y) not in closed:
          closed.append((x, y))
          succssorsPos = getSuccsorsPos((x, y), tunnels)
          for i in succssorsPos:
            if i not in closed:
              nextLength = length + 1
              dfs_stack.push((i, nextLength))
    return 0


   # get the closest food in tunnel using BFS search
  def getTunnelFood(self, gameState):
      

      curPos = gameState.getAgentState(self.index).getPosition()
      bfs_queue = util.Queue()
      closed = []
      bfs_queue.push(curPos)

      while not bfs_queue.isEmpty():
          x, y = bfs_queue.pop()
          if self.getFood(gameState)[int(x)][int(y)]:
              return (x, y)

          if (x, y) not in closed:
              closed.append((x, y))
              succssorsPos = getSuccsorsPos((x, y), tunnels)
              for i in succssorsPos:
                  if i not in closed:
                      bfs_queue.push(i)

      return None

    # get the rest steps for four agents
  def getTimeLeft(self, gameState):
      return gameState.data.timeleft

    
    # get all the legal position to jump to the middle boundary
  def getEntrance(self,gameState):
        width = gameState.data.layout.width
        height = gameState.data.layout.height
        legalPositions = [p for p in gameState.getWalls().asList(False)]
        legalRed = [p for p in legalPositions if p[0] == width / 2 - 1]
        legalBlue = [p for p in legalPositions if p[0] == width / 2]
        redEntrance = []
        blueEntrance = []
        for i in legalRed:
            for j in legalBlue:
                if i[0] + 1 == j[0] and i[1] == j[1]:
                    redEntrance.append(i)
                    blueEntrance.append(j)
        if self.red:
            return redEntrance
        else:
            return blueEntrance

    
    # sub method of MCTs. Simulate 20 steps using random walk and return the 
    # value of last state, will break if eaten by the ghost
  def OfsRollout(self,gameState):
    counter = 20
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghost = [a for a in enemies if not a.isPacman and a.getPosition() is not None]
    ghostPos = [a.getPosition() for a in ghost]
    curState = gameState
    while counter != 0:
      counter -= 1
      actions = curState.getLegalActions(self.index)
      nextAction = random.choice(actions)
      successor = self.getSuccessor(curState,nextAction)
      myPos = nextPos(curState.getAgentState(self.index).getPosition(),nextAction)
      if myPos in ghostPos:
          return -9999
      curState = successor
    return self.evaluate(curState,'Stop')


    # The function running MCTs. Looping time is 0.95 second 
  def simulation(self, gameState):
      (x1, y1) = gameState.getAgentPosition(self.index)
      root = Node((gameState, 0, 0))
      mct = Tree(root)
      startTime = time.time()
      while time.time() - startTime < 0.95:
          self.iteration(mct)
      nextState = mct.tree.chooseChild().value[0]
      (x2, y2) = nextState.getAgentPosition(self.index)
      if x1 + 1 == x2:
          return Directions.EAST
      if x1 - 1 == x2:
          return Directions.WEST
      if y1 + 1 == y2:
          return Directions.NORTH
      if y1 - 1 == y2:
          return Directions.SOUTH
      return Directions.STOP

    
    # iteration the tree one time: selection -> expand -> rollout -> back-propagation
  def iteration(self, mct):
      if mct.tree.children == []:
          self.expand(mct, mct.tree)
      else:
          leaf = mct.select()
          if leaf.value[2] == 0:
              r = self.OfsRollout(leaf.value[0])
              mct.backPropagate(r, leaf)
          elif leaf.value[2] == 1:
              self.expand(mct, leaf)
              newLeaf = random.choice(leaf.children)
              r = self.OfsRollout(newLeaf.value[0])
              mct.backPropagate(r, newLeaf)

    
    # sub funtion for MCTs, to expand a visited leaf node 
  def expand(self, mct, node):
      actions = node.value[0].getLegalActions(self.index)
      actions.remove(Directions.STOP)
      for action in actions:
          successor = node.value[0].generateSuccessor(self.index, action)
          successorNode = Node((successor, 0, 0))
          mct.insert(node, successorNode)


class OffensiveReflexAgent(ReflexCaptureAgent):


  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()                                             
    successor = self.getSuccessor(gameState, action)                          # successor state's game state
    curPos = gameState.getAgentState(self.index).getPosition()                # the current position of offensive agent
    myPos = successor.getAgentState(self.index).getPosition()                 # the successor state's position of offensive agent (if next step will die this will be startpoint)
    nextPosition = nextPos(curPos,action)                                     # the next position of offensive agent (if next step will die this will be next position)
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghost = [a for a in enemies if not a.isPacman and a.getPosition() is not None and manhattanDist(curPos,a.getPosition()) <= 5]    # only count the ghost near offensive agent
    scaredGhost = [a for a in ghost if a.scaredTimer > 1]                                     # the ghost that is scared but rest scared time > 1
    activeGhost = [a for a in ghost if a not in scaredGhost]                                  # the ghost that is not scared or almost be active (rest scared time < 1)
    invaders = [a for a in enemies if a.isPacman and a.getPosition() is not None]             # the invaders are Pacman found in our area
    currentFoodList = self.getFood(gameState).asList()                                        # store all the current dots
    openRoadFood = [a for a in currentFoodList if a not in tunnels]                           # store all the dots that are not in tunnel
    tunnelFood = [a for a in currentFoodList if a in tunnels]                                 # store all the dots that are in tunnel
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]     # store the reverse action of pre action
    capsule = self.getCapsules(gameState)                                                     # store all the current captures
    checkTunnel = self.ifWasteTunnel(gameState, successor)                                    # everytime check if agent in tunnel entry and evaluate that tunnel

    features['successorScore'] = self.getScore(successor)

    # if no ghost nearby, set these to None, only focus on eating closest dots
    if len(ghost) == 0:    
        self.capsule = None
        self.nextOpenFood = None
        self.nextTunnelFood = None

    # if agent has already enter opposite area, change to False
    if gameState.getAgentState(self.index).isPacman:  
        self.changeEntrance = False

    # calculate the carried dot of agent, change to 0 if come back
    if nextPosition in currentFoodList:                      
        self.carriedDot += 1
    if not gameState.getAgentState(self.index).isPacman:
        self.carriedDot = 0

    # if left time just for coming back, add this feature to force agent coming back
    if self.getTimeLeft(gameState)/4 < self.getLengthToHome(gameState) + 3: 
        features['distToHome'] = self.getLengthToHome(successor)
        return features

    # When no ghost nearby, the distance between Pacman with nearest food
    if len(activeGhost) == 0 and len(currentFoodList) != 0 and len(currentFoodList) >= 3:            
      features['safeFoodDist'] = min([self.getMazeDistance(myPos, food) for food in currentFoodList])
      if myPos in self.getFood(gameState).asList():
          features['safeFoodDist'] = -1

    # already can win, let agent safely comes back
    if len(currentFoodList) < 3:
      features['return'] = self.getLengthToHome(successor)  

    # The distance between Pacman with nearest active ghost 
    # adjusted to 100-distance to avoid negative number appeared 
    if len(activeGhost) > 0 and len(currentFoodList) >= 3:                                  
        dists = min([self.getMazeDistance(myPos, a.getPosition()) for a in activeGhost]) 
        features['distToGhost'] = 100 - dists
        ghostPos = [a.getPosition() for a in activeGhost]
        # When next position is ghost position or ghost neighbor positions
        if nextPosition in ghostPos:
            features['die'] = 1           
        if nextPosition in [getSuccsorsPos(p,legalPositions) for p in ghostPos][0]:
            features['die'] = 1
        # The distance between Pacman with nearest open road food
        if len(openRoadFood) > 0:             
            features['openRoadFood'] = min([self.getMazeDistance(myPos, food) for food in openRoadFood]) 
            if myPos in openRoadFood:
              features['openRoadFood'] = -1
        elif len(openRoadFood) == 0:
            features['return'] = self.getLengthToHome(successor)
          
    # to find all safe foods that are not in tunnel, and get the nearest safe food
    if len(activeGhost) > 0 and len(currentFoodList) >= 3:  
        if len(openRoadFood) > 0:
            safeFood = []
            for food in openRoadFood:
                if self.getMazeDistance(curPos, food) < min([self.getMazeDistance(a.getPosition(), food) for a in activeGhost]):
                    safeFood.append(food)
            if len(safeFood) != 0:
                closestSFdist = min([self.getMazeDistance(curPos, food) for food in safeFood])
                for food in safeFood:
                    if self.getMazeDistance(curPos, food) == closestSFdist:
                        self.nextOpenFood = food
                        break

    # to find all safe foods that are in tunnel, and get the nearest safe food
    if len(activeGhost) > 0 and len(tunnelFood) > 0 and len(scaredGhost) == 0 and len(currentFoodList) >= 3:
        minTFDist = min([self.getMazeDistance(curPos, tf) for tf in tunnelFood])
        safeTfood = []
        for tf in tunnelFood:
            tunnelEntry = getTunnelEntry(tf,tunnels,legalPositions)
            if self.getMazeDistance(curPos, tf) + self.getMazeDistance(tf, tunnelEntry) < min([self.getMazeDistance(a.getPosition(), tunnelEntry) for a in activeGhost]):
                safeTfood.append(tf)
        if len(safeTfood) > 0:
            closestTFdist = min([self.getMazeDistance(curPos, food) for food in safeTfood])
            for food in safeTfood:
                if self.getMazeDistance(curPos, food) == closestTFdist:
                    self.nextTunnelFood = food
                    break

    # force Pacman run to the nearest safe food
    if self.nextOpenFood != None:
        features['goToSafeFood'] = self.getMazeDistance(myPos, self.nextOpenFood)
        if myPos == self.nextOpenFood:
            features['goToSafeFood'] = -0.0001
            self.nextOpenFood = None

    if features['goToSafeFood'] == 0 and self.nextTunnelFood != None:
        features['goToSafeFood'] = self.getMazeDistance(myPos, self.nextTunnelFood)
        if myPos == self.nextTunnelFood:
            features['goToSafeFood'] = 0
            self.nextTunnelFood = None

    # find the nearest safe capsule if activeghost nearby
    if len(activeGhost) > 0 and len(capsule) != 0:
        for c in capsule:
            if self.getMazeDistance(curPos, c) < min([self.getMazeDistance(c, a.getPosition()) for a in activeGhost]):
                self.capsule = c

    if len(scaredGhost) > 0 and len(capsule) != 0:
        for c in capsule:
            if self.getMazeDistance(curPos, c) >= scaredGhost[0].scaredTimer and self.getMazeDistance(curPos, c) < min([self.getMazeDistance(c, a.getPosition()) for a in scaredGhost]):
                self.capsule = c

    if curPos in tunnels:
        for c in capsule:
            if c in getATunnel(curPos,tunnels):
                self.capsule = c

    # if find the nearest safe capsule, and agent is chased by ghost, run to that capsule
    if self.capsule != None:
        features['distanceToCapsule'] = self.getMazeDistance(myPos, self.capsule)
        if myPos == self.capsule:
            features['distanceToCapsule'] = 0
            self.capsule = None

    # if no active ghost nearby, agent will not eat that capsule if it doesn't block the way
    if len(activeGhost) == 0 and myPos in capsule:
        features['leaveCapsule'] = 0.1

    # Normally no use, the feature appears when Pacman needs to stop to let the ghost walk one 
    # step, in order to avoid death.
    if action == Directions.STOP: features['stop'] = 1

    # When in a tunnel entrance, this feature force Pacman does not go into an empty tunnel
    if successor.getAgentState(self.index).isPacman and curPos not in tunnels and \
        successor.getAgentState(self.index).getPosition() in tunnels and checkTunnel == 0:
        features['noFoodTunnel'] = -1

    # When ghost nearby and Pacman in a tunnel entrance, and this tunnel has food 
    # inside, it will calculate the distance to judge if it can eat food then leave 
    # tunnel safely. If not, this feature will appear to force it to leave the tunnel
    if len(activeGhost) > 0:
         dist = min([self.getMazeDistance(curPos, a.getPosition()) for a in activeGhost])
         if checkTunnel != 0 and checkTunnel*2 >= dist-1:
             features['wasteAction'] = -1

    if len(scaredGhost) > 0:
         dist = min([self.getMazeDistance(curPos, a.getPosition()) for a in scaredGhost])
         if checkTunnel != 0 and checkTunnel*2 >= scaredGhost[0].scaredTimer -1:
             features['wasteAction'] = -1

    # When Pacman in a tunnel and suddenly ghost found nearby, Pacman will judge when it 
    # should leave this tunnel
    if curPos in tunnels and len(activeGhost) > 0:
        foodPos = self.getTunnelFood(gameState)
        if foodPos == None:
            features['escapeTunnel'] = self.getMazeDistance(nextPos(curPos,action), self.tunnelEntry)
        else:
            lengthToEscape = self.getMazeDistance(myPos, foodPos) + self.getMazeDistance(foodPos, self.tunnelEntry)
            ghostToEntry = min([self.getMazeDistance(self.tunnelEntry, a.getPosition()) for a in activeGhost])
            if ghostToEntry - lengthToEscape <= 1 and len(scaredGhost) == 0:
                features['escapeTunnel'] = self.getMazeDistance(nextPos(curPos,action), self.tunnelEntry)

    if curPos in tunnels and len(scaredGhost) > 0:
        foodPos = self.getTunnelFood(gameState)
        if foodPos == None:
            features['escapeTunnel'] = self.getMazeDistance(nextPos(curPos,action), self.tunnelEntry)
        else:
            lengthToEscape = self.getMazeDistance(myPos, foodPos) + self.getMazeDistance(foodPos, self.tunnelEntry)
            if  scaredGhost[0].scaredTimer - lengthToEscape <= 1:
                features['escapeTunnel'] = self.getMazeDistance(nextPos(curPos,action), self.tunnelEntry)

    if not gameState.getAgentState(self.index).isPacman and len(activeGhost) > 0 and self.stuckStep != -1:
        self.stuckStep += 1

    if gameState.getAgentState(self.index).isPacman or myPos == self.nextEntrance:
        self.stuckStep = 0
        self.nextEntrance = None

    if self.stuckStep > 10:
        self.stuckStep = -1
        self.nextEntrance = random.choice(self.getEntrance(gameState))   

    # When Pacman get stuck with the ghost between the boundary, after 10 steps, 
    # this feature will appear to force Pacman change another random selected entrance
    if self.nextEntrance != None and features['goToSafeFood'] == 0:
        features['runToNextEntrance'] = self.getMazeDistance(myPos,self.nextEntrance)

    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore':1, 'distToHome':-100, 'safeFoodDist':-2, 'openRoadFood' :-3,'distToGhost': -10, 'die':-1000,'goToSafeFood': -11,'distanceToCapsule': -1200, 
          'return':-1,'leaveCapsule':-1, 'stop':-50, 'noFoodTunnel':100,'wasteAction': 100,'escapeTunnel':-1001,'runToNextEntrance':-1001}


  def getLengthToHome(self, gameState):  
      curPos = gameState.getAgentState(self.index).getPosition()
      width = gameState.data.layout.width
      height = gameState.data.layout.height
      legalPositions = [p for p in gameState.getWalls().asList(False)]
      legalRed = [p for p in legalPositions if p[0] == width / 2 - 1]
      legalBlue = [p for p in legalPositions if p[0] == width / 2]
      if self.red:
          return min([self.getMazeDistance(curPos, a) for a in legalRed])
      else:
          return min([self.getMazeDistance(curPos, a) for a in legalBlue])


class DefensiveReflexAgent(ReflexCaptureAgent):

    
    # getLengthToBoundary: return the closest boundary position to agent
  def getLengthToBoundary(self, gameState):     
      curPos = gameState.getAgentState(self.index).getPosition()
      width = gameState.data.layout.width
      height = gameState.data.layout.height
      legalPositions = [p for p in gameState.getWalls().asList(False)]
      legalRed = [p for p in legalPositions if p[0] == width / 2 - 1]
      legalBlue = [p for p in legalPositions if p[0] == width / 2]
      if self.red:
          return min([self.getMazeDistance(curPos, a) for a in legalRed])
      else:
          return min([self.getMazeDistance(curPos, a) for a in legalBlue])

  def getFeatures(self, gameState, action):

    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    curPos = gameState.getAgentState(self.index).getPosition()   # current defensive agent position
    curState = gameState.getAgentState(self.index)               # current defensive agent state
    sucState = successor.getAgentState(self.index)               # next defensive agent state
    sucPos = sucState.getPosition()                              # successor state's position of agent
    curCapsule = self.getCapsulesYouAreDefending(gameState)      # current capsule
    lengthToBoundary = self.getLengthToBoundary(successor)       # length to the nearest boundary position
  
    # This feature force our defensive agent only walks in defensive area
    features['onDefense'] = 100
    if sucState.isPacman: features['onDefense'] = 0

    # At the beginning , our defensive agent will run to boundary as quick as possible
    if self.runToBoundary == None:
        features['runToBoundary'] = self.getLengthToBoundary(successor)

    if self.getLengthToBoundary(successor) <= 2:
        self.runToBoundary = 0


    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]       # successor state's enemies
    curEnemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]    # current enemies
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]          # successor state's enemies
    curInvaders = [a for a in curEnemies if a.isPacman and a.getPosition() != None]    # current invaders


    # When the ghost can block tunnel to get Pacman stuck, this feature forces 
    # agent to run to that tunnel entry
    if self.invadersGuess:
        self.enemyGuess.observe(self, gameState)
        enemyPos = self.enemyGuess.getPossiblePosition(curInvaders[0])
        features['runToTunnelEntry'] = self.getMazeDistance(enemyPos, sucPos)
        self.enemyGuess.elapseTime()

    if self.ifNeedsBlockTunnel(curInvaders, curPos, curCapsule) and curState.scaredTimer == 0:
        features['runToTunnelEntry'] = self.getMazeDistance(getTunnelEntry(curInvaders[0].getPosition(),tunnels,legalPositions),sucPos)
        return features

    # When in the tunnel and no invaders nearby, it will leave this tunnel as 
    # quick as possible
    if curPos in defensiveTunnels and len(curInvaders) == 0:
        features['leaveTunnel'] = self.getMazeDistance(self.start, sucPos)

    # This feature will let the ghost try to kill the Pacman
    features['numInvaders'] = len(invaders) 

    # This feature forces the ghost does not go into a tunnel when no invader found       
    if len(curInvaders) == 0 and not successor.getAgentState(self.index).isPacman and curState.scaredTimer == 0:
        if  curPos not in defensiveTunnels and successor.getAgentState(self.index).getPosition() in defensiveTunnels: 
            features['wasteAction'] = -1

    # features['invaderDistance']: The distance between my ghost with the nearest invader
    # features['lengthToBoundary']: length to the nearest boundary position
    # This feature will appear when ghost chasing the Pacman, to ensure ghost 
    # avoids Pacman coming back to home when chasing
    if len(invaders) > 0 and curState.scaredTimer == 0:            
        dists = [self.getMazeDistance(sucPos, a.getPosition()) for a in invaders]
        features['invaderDistance'] = min(dists)
        features['lengthToBoundary'] = self.getLengthToBoundary(successor)
    
    # When ghost is in scared time, it will try to always keep two distance far away from the Pacman
    if len(invaders) > 0 and curState.scaredTimer != 0:           
        dists = min([self.getMazeDistance(sucPos, a.getPosition()) for a in invaders])
        features['followMode'] = (dists-2)*(dists-2)
        if curPos not in defensiveTunnels and successor.getAgentState(self.index).getPosition() in defensiveTunnels:
            # This feature forces the ghost does not go into a tunnel when no invader found
            features['wasteAction'] = -1

    # Distance between agent and capsule, if there are invaders nearby
    if len(invaders) > 0 and len(curCapsule) != 0:         
        dist2 = [self.getMazeDistance(c, sucPos) for c in curCapsule]
        features['protectCapsules'] = min(dist2)

    # When the ghost can block tunnel to get Pacman stuck, this feature forces agent stopping
    if action == Directions.STOP: features['stop'] = 1  

    # This feature let our ghost do not go reverse      
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1  

    # When ghost found our food get lost, it will run to that lost food place
    if self.getPreviousObservation() != None:
      if len(invaders) == 0 and self.ifLostFood() != None:
          self.curLostFood = self.ifLostFood()

      if self.curLostFood != None and len(invaders) == 0: 
          features['goToLostFood'] = self.getMazeDistance(sucPos,self.curLostFood)
      
      if sucPos == self.curLostFood or len(invaders) > 0:
          self.curLostFood = None

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -100, 'onDefense': 10, 'invaderDistance': -10, 'stop': -100, 'reverse': -2,'lengthToBoundary':-3,
     'protectCapsules': -3, 'wasteAction':200,'followMode':-100, 'runToTunnelEntry': -10, 'leaveTunnel':-0.1,'runToBoundary':-2,'goToLostFood':-1}

  """
  This function is used to check if our agent needs to block the tunnel,
  True means need to block.
  """
  def ifNeedsBlockTunnel(self, curInvaders, currentPostion, curCapsule): 
    if len(curInvaders) == 1:
      invadersPos = curInvaders[0].getPosition()
      if invadersPos in tunnels:
        tunnelEntry = getTunnelEntry(invadersPos, tunnels, legalPositions)
        if self.getMazeDistance(tunnelEntry,currentPostion) <= self.getMazeDistance(tunnelEntry,invadersPos) and curCapsule not in getATunnel(invadersPos,tunnels):
           return True
    return False


  """
  This function is used to check if any of our foods lost currently
  """
  def ifLostFood(self):
        preState = self.getPreviousObservation()
        currState = self.getCurrentObservation()
        myCurrFood = self.getFoodYouAreDefending(currState).asList()
        myLastFood = self.getFoodYouAreDefending(preState).asList()
        if len(myCurrFood) < len(myLastFood):
            for i in myLastFood:
                if i not in myCurrFood:
                    return i
        return None
