def read_file(file_name):
  """ Reads the contents of a file.

  Args:
      file_name: The name of the file to read.

  Returns:
      str: The contents of the file.
  """
  with open(file_name, 'r') as f:
    contents = f.read()
  return contents

  # Alternatively, to read line by line:
  # with open(file_name, 'r') as f:
  #   for line in f:
  #     print(line, end='')  # Print without adding newline after each line

def read_file_into_list(file_name):
  """ Reads a file and stores each line as an element in a list.

  Args:
      file_name: The name of the file to read.

  Returns:
      list: A list where each element is a line from the file.
  """
  lines = []
  with open(file_name, 'r') as f:
    for line in f:
      lines.append(line.strip())  # Remove trailing newline characters
  return lines

def write_first_line_to_file(file_contents, output_filename):
  """ Writes the first line of a string to a file.

  Args:
      file_contents: The string containing the content to write.
      output_filename: The name of the file to write to.
  """
  first_line = file_contents.splitlines()[0]  # Get the first line
  with open(output_filename, 'w') as f:
    f.write(first_line)

def read_even_numbered_lines(file_name):
  """ Reads the even-numbered lines of a file.

  Args:
      file_name: The name of the file to read.

  Returns:
      list: A list containing the even-numbered lines of the file.
  """
  even_lines = []
  with open(file_name, 'r') as f:
    for i, line in enumerate(f):
      if i % 2 == 0:  # Check if line index is even
        even_lines.append(line.strip())
  return even_lines

def read_file_in_reverse(file_name):
  """ Reads a file and returns a list of the lines in reverse order.

  Args:
      file_name: The name of the file to read.

  Returns:
      list: A list containing the lines of the file in reverse order.
  """
  lines = []
  with open(file_name, 'r') as f:
    for line in f:
      lines.insert(0, line.strip())  # Insert lines at the beginning to reverse
  return lines

# Sample usage (commented out)
def main():
  file_contents = read_file("sampletext.txt")
  print(read_file_into_list("sampletext.txt"))
  write_first_line_to_file(file_contents, "online.txt")

if __name__ == "__main__":
  main()
