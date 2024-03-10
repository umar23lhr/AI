# Import statements
import json
from employee import employee_name , age , title, details

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


    

def write_json_to_file(json_obj, output_file):
    """Write json string to file.

    Args:
        json_obj (str): Json string containing employee information
        output_file (str): The file the json is being written to
    """
    with open(output_file, 'w') as newfile:
        newfile.write(json_obj)
        
    

def main():

    details()

    employee_dict = create_dict(employee_name, age, title)

    json_object = json.dumps(employee_dict, indent=2)

    write_json_to_file(json_object, "employee_details.json")

if __name__ == "__main__":
    main()

