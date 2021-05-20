from recommender_engine import Recommender_Engine

print('Welcome to the restaurants recommender')
print('\n Loading...')

boolean = True
personalized = False
stars_original = False
n = 5
recommender = Recommender_Engine(personalized=True)

print('Recommender Engine is ready to use!')


# Obtain keyword
def get_keyword():
    city, zipcode, state = None, None, None
    max_distance = 10
    cuisine, style, price = None, None, None

    cuisines = ['american (new)', 'american (traditional)', 'asian fusion', 'bagels', 'bakeries',
                'barbeque', 'beer', 'bubble tea', 'burgers', 'cajun', 'chicken wings', 'chinese',
                'coffee & tea', 'cupcakes', 'custom cakes', 'desserts', 'donuts', 'ethnic food',
                'french', 'gluten-free', 'greek', 'hawaiian', 'hot dogs', 'ice cream & frozen yogurt',
                'indian', 'italian', 'japanese', 'korean', 'latin american', 'local flavor', 'mediterranean',
                'mexican', 'middle eastern', 'pizza', 'salad', 'sandwiches', 'seafood', 'shaved ice', 'soup',
                'southern', 'specialty food', 'steakhouse', 'sushi bars', 'tacos', 'tapas/small plates', 'tex-mex',
                'thai', 'vegan', 'vegetarian', 'vietnamese', 'wine & spirits']

    styles = ['bars', 'beer bars', 'breakfast & brunch', 'breweries', 'buffets', 'cafes', 'casinos',
              'caterers', 'cocktail bars', 'dance clubs', 'delis', 'diners', 'dive bars', 'fast food',
              'food delivery services', 'food stands', 'food trucks', 'juice bars & smoothies', 'lounges',
              'music venues', 'nightlife', 'performing arts', 'pubs', 'restaurants', 'sports bars', 'street vendors',
              'wine bars']

    user_input = input('Filter by? \n1. Location(city, zipcode, state);\n2. Cuisine; '
                       '\n3. Style; \n4. Price range. Please separate answer by comma \n')
    if len(user_input) > 0:
        print('Amazing! Lets gather filtering criteria')
        keyword = user_input.split(',')
        for i in keyword:
            try:
                i = int(i)
            except:
                print("Oops, invalid input of '{}' skipped".format(i))
                continue
            if i == 1:
                print('Enter location of interest or use ENTER key to skip')
                user_input = input('Please enter city of interest or use ENTER key to skip\n')
                if len(user_input) > 0:
                    city = user_input
                user_input = input('Please enter zipcode of interest or use ENTER key to skip\n')
                if len(user_input) > 0:
                    zipcode = user_input
                user_input = input('Please enter state of interest or use ENTER key to skip\n')
                if len(user_input) > 0:
                    state = user_input
                user_input = input('Please enter maximum distance or use ENTER key to skip\n')
                if len(user_input) > 0:
                    try:
                        max_distance = int(user_input)
                    except:
                        print('Oops, invalid distance. Maximum distance is now set to 10 miles')

            elif i == 2:
                while True:
                    user_input = input('Please select cuisine of interest or user ENTER to skip:\n{\n'.format(cuisines))
                    if len(user_input) > 0:
                        if user_input in cuisines:
                            cuisine = user_input
                            break
                        else:
                            print('Oops, invalid cuisine')
                    else:
                        break

            elif i == 3:
                while True:
                    user_input = input('Please select style of interest or user ENTER to skip:\n{}\n'.format(cuisines))
                    if len(user_input) > 0:
                        if user_input in styles:
                            style = user_input
                            break
                        else:
                            print('Oops, invalid style')
                    else:
                        break

            elif i == 4:
                user_input = input('Please indicate your price range of interest: \n1. Cheap ($);\n2. Medium ($$);'
                                   '\n3. Expensive ($$$);\n4. Most expensive($$$$)'
                                   '\nPlease enter the corresponding number(s) separated by comma\n')
                if len(user_input) > 0:
                    price = user_input

            else:
                print("Oops, invalid category of '{}' skipped".format(i))

    return city, zipcode, state, max_distance, cuisine, style, price


# UI of Recommendation Engine
while boolean:

    user_input = input('Would you like to have personalized recommendations? Yes/No \n')
    if user_input.startswith('Y') or user_input.startswith('y'):
        personalized = True

    # Personalized module
    if personalized:
        print('Lets get your personalized recommendations')

        user_input = input('Please enter Yelp user_id: \n')
        if len(user_input) == 0:
            print('No user_id provided, try again')
            continue
        elif len(user_input) != 22:
            print('Invalid user_id, try again')
            continue
        else:
            user_id = user_input
            print('Amazing, valid user_id!')
            user_input = input('Which kind of personalization you prefer?'
                               '\n1. Something new based on people like you?'
                               '\n2. Something similar to your past favorites?'
                               '\n Please enter 1 or 2\n')
            try:
                user_input = int(user_input)
                if user_input not in [1, 2]:
                    print('Invalid input, tray again')
                    continue
                else:
                    print('Awesome, all set')
                    if user_input == 1:
                        print('**********************************************************')
                        result = recommender.collaborative_filtering(user_id=user_id)
                        print('**********************************************************')
                    else:
                        print('**********************************************************')
                        result = recommender.content_filtering(user_id=user_id)
                        print('**********************************************************')
            except:
                print('Oops, invalid input. Try again')
                continue

    # Non personalized module
    elif user_input.startswith('N') or user_input.startswith('n'):

        print('Lets filter by keyword and get recommendations')
        city, zipcode, state, max_distance, cuisine, style, price = get_keyword()
        user_input = input('Want to rank restaurants by smart ratings? Yes/No')
        if user_input.startswith('N') or user_input.startswith('n'):
            stars_original = True
        print('Here are your recommendations')
        print('**********************************************************')
        result = recommender.keyword_filtering(city=city, zipcode=zipcode, state=state, distance_max=max_distance,
                                               cuisine=cuisine, style=style, price=price, stars_original=stars_original)
        print('**********************************************************')

    if result is not None and len(result) > 0:
        user_input = input('How many recommendations would you like to see?')
        try:
            n = int(user_input)
            print('**********************************************************')
            recommender.display(n=n)
            print('**********************************************************')
        except:
            pass
        user_input = input('Would you like to further filter? Yes/No')
        if user_input.startswith('Y') or user_input.startswith('y'):
            city, zipcode, state, max_distance, cuisine, style, price = get_keyword()
            print('**********************************************************')
            result = recommender.keyword_filtering(catalog=result, city=city, zipcode=zipcode, state=state,
                                                   distance_max=max_distance, cuisine=cuisine, style=style,
                                                   price=price, personalized=personalized)
            print('**********************************************************')

    # Restart or quit
    print('All done!')
    user_input = input('Please enter q to quit the engine or r to restart')
    if len(user_input) == 0 or user_input.startswith('Q') or user_input.startswith('Q'):
        boolean = False
        print('Thank you, see you next time!')