# Outer loop iterates through numbers 1 to 10
for num in range(1, 11):
  # Inner loop iterates through numbers 1 to 10 for each outer loop iteration
  for i in range(1, 11):
    # Calculate and print the product
    product = num * i
    print(f"{num} x {i} = {product}")
  # Add an empty line after each table for better formatting
  print()
