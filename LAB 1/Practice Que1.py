# Get user input with error handling
try:
  length = float(input("Enter the length of the rectangle: "))
  width = float(input("Enter the width of the rectangle: "))
except ValueError:
  print("Invalid input. Please enter numbers only.")

# Calculate the area
area = length * width

# Print the area with a complete sentence
print(f"The area of the rectangle is {area:.2f}.")
