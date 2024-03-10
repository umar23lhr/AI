# Import statements
import json
import employee

def create_dict(name, age, title):
    """Creates a dictionary that stores an employee's information.

    Args:
        name (str): Name of employee
        age (int): Age of employee
        title (str): Title of employee

    Returns:
        dict: A dictionary that maps "first_name", "age", and "title" to the
              name, age, and title arguments, respectively. Make sure that 
              the values are typecasted correctly (name - string, age - int, 
              title - string)
    """
    employee_dict = {
        "first_name": str(name),
        "age": int(age),
        "title": str(title)
    }
    return employee_dict


    

def write_json_to_file():
    """ Write json string to file

    1. Open a new file defined by output_file
    2. Write json_obj to the new file

    Args:
        json_obj: json string containing employee information
        output_file: the file the json is being written to
     
    WRITE YOUR SOLUTION BELOW
    """

    

def main():
    # Print the contents of details() -- This should print the details of an employee
    

    # Create employee dictionary
   

    # Use a function called dumps from the json module to convert employee_dict
    # into a json string and store it in a variable called json_object.
    

    # Write out the json object to file
    

if __name__ == "__main__":
    main()
