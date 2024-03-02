menu = {
    1: {"name": 'espresso',
        "price": 1.99},
    2: {"name": 'coffee', 
        "price": 2.50},
    3: {"name": 'cake', 
        "price": 2.79},
    4: {"name": 'soup', 
        "price": 4.50},
    5: {"name": 'sandwich',
        "price": 4.99}
}

def display_menu():
    print("Menu:")
    for item_number, details in menu.items():
        print(f"{item_number}. {details['name']} - ${details['price']}")


def select_items():
    order = []
    for _ in range(3):
        try:
            item_number = int(input("Enter the item number you want to select: "))
            order.append(menu[item_number])
        except (ValueError, KeyError):
            print("Invalid input. Please enter a valid item number.")
    return order



def calculate_subtotal(order):
    """Calculates the subtotal of an order.

    Args:
        order: List of dictionaries containing item details.

    Returns:
        The total price of all items in the order.
    """
    print('Calculating bill subtotal...')
    subtotal = sum(item for item in order)
    return subtotal

def calculate_tax(subtotal):
    """Calculates the tax for an order.

    Args:
        subtotal: The total price of the order.

    Returns:
        The tax amount based on a 15% tax rate, rounded to two decimals.
    """
    print('Calculating tax from subtotal...')
    tax_rate = 0.15
    tax = round(subtotal * tax_rate, 2)
    return tax



def summarize_order(order):
    """Summarizes the order details.

    Args:
        order: List of dictionaries containing item details.

    Returns:
        A tuple containing a list of item names and the total order price.
    """
    print('Summarizing order...')
    names = [item['name'] for item in order]
    subtotal = calculate_subtotal(order)
    tax = calculate_tax(subtotal)
    total = round(subtotal + tax, 2)
    return names, total

def main():
    """Displays the menu, allows item selection, and summarizes the order."""
    display_menu()
    order = select_items()
    names, total = summarize_order(order)

    print("\nYour order summary:")
    for name in names:
        print(f"- {name}")
    print(f"Total: ${total:.2f}")


if __name__ == "__main__":
    main()
