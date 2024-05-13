try:
  x = int(input("Enter a number: "))
  y = 10 / x
  print("Result:", y)
except ValueError:
  print("Please enter a valid integer.")
except ZeroDivisionError:
  print("Cannot divide by zero.")
except Exception as e:
  print("An error occurred:", e)
# ---------------------------------------------
  with open("sample.txt", "w") as file:
    file.write("Hello, World!\n")
    file.write("This is a sample file.\n")

#Reading from a file

  with open("sample.txt", "r") as file:
    contents = file.read()
    print("File Contents:")
    print(contents)

#Storing file contents in a list

  with open("sample.txt", "r") as file:
    lines = file.readlines()
    print("File Contents as List:")
    print(lines)

# ----------------------------------------------
def add(a, b):

    return a + b

#Recursive function to calculate factorial

def factorial(n): 
    if n == 0:
     return 1
    else:

     return n * factorial(n-1)

#Map function to double each element in a list

numbers = [1, 2, 3, 4, 5]
doubled_numbers = list(map(lambda x: x*2, numbers)) 
print("Doubled Numbers:", doubled_numbers)

#Filter function to filter even numbers from a list

even_numbers = list(filter(lambda x: x%2==0, numbers)) 
print("Even Numbers:", even_numbers)

#List comprehension to generate squares of numbers

squares =[x** 2 for x in numbers] 
print("Squares:", squares)
# ---------------------------------------------

class Person:
 def __init__(self, name): 
  self.name = name

#Instantiate a custom object 
person1 = Person("Alice")
# ---------------------------------------------

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age =age

#3. Instantiate a Custom Object

person1 = Person("Alice", 30)

#4. Instance Methods

class Dog:
    def __init__(self, name):
        self.name = name
    def bark(self):
        return f"(self.name) says woof!"
dog1=Dog("Buddy")
print(dog1.bark())

# ----------------------------------------

class Animal:
    def speak(self):
        raise NotImplementedError("Subclass must implement abstract method")

class Dog(Animal):
    def speak(self):
        return "Woof!"
    
# -----------------------------------------

class Animal:
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        return "Woof!"

class Cat(Animal):
    def make_sound(self):
        return "Meow!"

class DogCat(Dog, Cat):
    pass

pet=DogCat() 
print(pet.make_sound()) # Output: Woof!

# -------------------------------------------------
from abc import ABC, abstractmethod

class Vehicle(ABC): 
    @abstractmethod 
    def move(self):
        pass

class Car(Vehicle):
    def move(self):
        return "Car is moving"
    # -------------------------------------

class A:
    def process(self):
        return "A"

class B(A):
    def process(self):
        return "B"

class C(A):
    def process(self):
        return "C"

class D(B, C):
    pass
print(D().process()) # Output: B
