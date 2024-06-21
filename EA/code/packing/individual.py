import matplotlib.pyplot as plt
import numpy as np
import random

class Circle:
    def __init__(self, r):
        self.r = r

class Rect:
    def __init__(self, x, y, width, height, value):
        self.y = y
        self.x = x
        self.width = width
        self.height = height
        self.value = value
        self.__vertices = None
        
    def __repr__(self):
        return f"Rect({self.width}, {self.height}, {self.value})"
    
    def vertices(self):
        # order according to the watch, starting from the top left corner
        return [(self.x - self.width / 2, self.y - self.height / 2), (self.x + self.width / 2, self.y - self.height / 2), (self.x + self.width / 2, self.y + self.height / 2), (self.x - self.width / 2, self.y + self.height/2)]
    
    def check_if_outside_circle(self, r):
        # check if the rect is outside the circle of given radius
        for x, y in self.vertices():
            if (x) ** 2 + (y) ** 2 > r ** 2:
                return True # is outside the circle
        return False
    
class Individual:
    def __init__(self, rects=[], r=1, timeout=1000):
        self.rects = rects
        self.r = r
        self.collisions = []
        self.modified_rects = True
        self.timeout = timeout
        
    def list_rectangles_outside_circle(self):
        bad_rects = []
        for rect in self.rects:
            if rect.check_if_outside_circle(self.r):
                bad_rects.append(rect)
        if len(bad_rects) > 0:
            return bad_rects
        return []
    
    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.add_patch(plt.Circle((0, 0), self.r, fill=False))
        for rect in self.rects:
            ax.add_patch(plt.Rectangle((rect.x - rect.width / 2, rect.y - rect.height / 2), rect.width, rect.height, fill=True, facecolor='blue', lw=1, edgecolor='black'))
        ax.set_xlim(-self.r * 1.1, self.r* 1.1)
        ax.set_ylim(-self.r* 1.1, self.r* 1.1)
        
        ax.set_title("Score: " + str(self.score()))
        ax.axis('off')
        return ax
        
    def list_collisions(self):
        if not self.modified_rects:
            return self.collisions
        self.collision = []
        self.rects.sort(key = lambda rect: (rect.x - rect.width / 2, rect.y - rect.height / 2))
        for i in range(len(self.rects)):
            for j in range(i + 1, len(self.rects)):
                if Individual.check_if_pair_collides(self.rects[i], self.rects[j]):
                    self.collision.append((self.rects[i], self.rects[j]))
                    
        return self.collision
    
    def check_if_pair_collides(rect1, rect2):
        # check if two rectangles collide
        
        if np.abs(rect1.x - rect2.x) <  (rect1.width  + rect2.width) / 2 and np.abs(rect1.y - rect2.y) < (rect1.height + rect2.height) / 2:
            return True
        
        return False
    
    def is_collision(self):
        return len(self.list_collisions()) > 0
    
    def is_any_outside_circle(self):
        return len(self.list_rectangles_outside_circle()) > 0
    
    def score(self):
        if self.is_collision() or self.is_any_outside_circle():
            return -1
        return sum([rect.value for rect in self.rects])
    
    def add_rect(self, rect):
        self.rects.append(rect)
        self.modified_rects = True
        
    def check_if_new_rect_will_collide(self, rect, existing = False):
        for r in self.rects:
            if Individual.check_if_pair_collides(r, rect):
                if existing and r == rect:
                    continue
                return True
        return False
    
    
        
    def generate_rect_randomly(self, width, height, value):
        r_rect = random.uniform(0, self.r - max(width, height) / 2)
        theta = random.uniform(0, 2 * np.pi)
        
        x = r_rect * np.cos(theta)
        y = r_rect * np.sin(theta)
        
        rect = Rect(x, y, width, height, value)
        if not rect.check_if_outside_circle(self.r) and not self.check_if_new_rect_will_collide(rect):
            self.add_rect(rect)
            self.modified_rects = False # no need to recompute the collisions
            return rect
        return None
    
    def generate_rect_from_data(self, rect_data, timeout=None, p = None):
        if timeout is None:
            timeout = self.timeout
        time = 0
        while True:
            i = np.random.choice(list(range(len(rect_data))), p=p)
            if time > timeout:
                return None
            rect = self.generate_rect_randomly(rect_data.iloc[i]["width"], rect_data.iloc[i]["height"], rect_data.iloc[i]["value"])
            if rect is not None:
                return rect
            time += 1
            
    def mutate(self, type = "move", axis = "both", how_many_move = 0.001, magnitude_move = 1, collide_strategy = "skip", minus_sign_prob = 0.2, p = None,
               rect_data=None, timeout=None):
        if timeout is None:
            timeout = self.timeout
        assert type in ["move", "replace", "add", "slide"]
        
        if type == "replace":
            assert rect_data is not None, "rect_data must be provided for replace mutation"
            
            rect_to_replace = random.choice(self.rects)
            self.rects.remove(rect_to_replace)
            self.modified_rects = True
            
            self.generate_rect_from_data(rect_data, timeout, p=p)
            
            return self
        elif type == "move":
            assert axis in ["both", "x", "y"], "axis must be one of both, x, y"
            assert 0 < how_many_move <= 1, "`how_many_move` is percentage of rectangles to move, must be between 0 and 1"
            assert magnitude_move > 0, "magnitude_move - sigma of gaussian noise must be positive"
            assert collide_strategy in ["skip", "remove"], "collide_strategy must be one of skip, remove"
                        
            # select subset of rectangles to move
            n_rects_to_move = int(len(self.rects) * how_many_move)
            rects_to_move = random.sample(self.rects, n_rects_to_move)
            
            for rect_to_move in rects_to_move:
                x_move = 0
                y_move = 0
                if axis == "both" or axis == "x":
                    x_move = random.gauss(0, magnitude_move)
                    rect_to_move.x += x_move
                
                if axis == "both" or axis == "y":
                    y_move = random.gauss(0, magnitude_move)
                    rect_to_move.y += y_move
                
                if rect_to_move.check_if_outside_circle(self.r) or self.check_if_new_rect_will_collide(rect_to_move, existing = True):
                    if collide_strategy == "remove":
                        self.rects.remove(rect_to_move)
                        self.modified_rects = True
                    elif collide_strategy == "skip":
                        rect_to_move.x -= x_move
                        rect_to_move.y -= y_move                   
        
        elif type == "add":
            assert rect_data is not None, "rect_data must be provided for add mutation"
            self.generate_rect_from_data(rect_data, timeout)   
            
        elif type == "slide":
            assert axis in ["both", "x", "y"], "axis must be one of both, x, y"
            assert 0 < how_many_move <= 1, "`how_many_move` is percentage of rectangles to move, must be between 0 and 1"
            assert magnitude_move > 0, "magnitude_move - sigma of gaussian noise must be positive"
            
            if axis == "both":
                axis = ["x", "y"]
            
            # select subset of rectangles to move
            n_rects_to_move = int(len(self.rects) * how_many_move)
            rects_to_move = random.sample(self.rects, n_rects_to_move)
            
            for rect_to_move in rects_to_move:
                for ax in axis:
                    collided = False
                    if random.uniform(0, 1) > minus_sign_prob:
                        sign = 1
                    else:
                        sign = -1
                    while not collided:
                        move = np.abs(random.gauss(0, magnitude_move)) * sign
                        if ax == "x":
                            rect_to_move.x += move
                        elif ax == "y":
                            rect_to_move.y += move
                
                        if rect_to_move.check_if_outside_circle(self.r) or self.check_if_new_rect_will_collide(rect_to_move, existing = True):
                            collided = True
                            if ax == "x":
                                rect_to_move.x -= move
                            elif ax == "y":
                                rect_to_move.y -= move     
            
        
        assert self.is_any_outside_circle() == False, "There are rectangles outside the circle"      
        
        return self
    
    @staticmethod
    def crossover(i1, i2, axis = "x", add_new_rects = False, add_new_rects_iterations = 10, rect_data = None):
        assert axis in ["x", "y"], "axis must be one of x, y"
        assert i1.r == i2.r, "Circles of the individuals must have the same radius"
        
        # select randomly a line to split the rectangles
        
        new_i1 = Individual(rects=[], r=i1.r, timeout=i1.timeout)
        new_i2 = Individual(rects=[], r=i1.r, timeout=i2.timeout)
        
        split = random.uniform(-i1.r, i1.r)
        for rect in i1.rects:
            new_rect = Rect(rect.x, rect.y, rect.width, rect.height, rect.value)
            if axis == "x":
                if rect.x + rect.width / 2 <= split:
                    new_i1.rects.append(new_rect)
                if rect.x - rect.width / 2 >= split:
                    new_i2.rects.append(new_rect)
            elif axis == "y":
                if rect.y + rect.height / 2 <= split:
                    new_i1.rects.append(new_rect)
                if rect.y - rect.height / 2 >= split:
                    new_i2.rects.append(new_rect)
        
        for rect in i2.rects:
            new_rect = Rect(rect.x, rect.y, rect.width, rect.height, rect.value)
            if axis == "x":
                if rect.x + rect.width / 2 <= split:
                    new_i2.rects.append(new_rect)
                if rect.x - rect.width / 2 >= split:
                    new_i1.rects.append(new_rect)
            elif axis == "y":
                if rect.y + rect.height / 2 <= split:
                    new_i2.rects.append(new_rect)
                if rect.y - rect.height / 2 >= split:
                    new_i1.rects.append(new_rect)
            
        
        new_i1.modified_rects = True
        new_i2.modified_rects = True
        
        if add_new_rects:
            assert rect_data is not None, "rect_data must be provided for add_new_rects"
            for _ in range(add_new_rects_iterations):
                new_i1.generate_rect_from_data(rect_data)
                new_i2.generate_rect_from_data(rect_data)
        
        return new_i1, new_i2
        
        