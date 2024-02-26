try:
  # Get user input with error handling
  number = int(input("Enter an integer: "))

  # Check if the number is positive, negative, or zero
  if number > 0:
    print("The number is positive.")
  elif number < 0:
    print("The number is negative.")
  else:
    print("The number is zero.")
except ValueError:
  print("Invalid input: Please enter an integer.")
