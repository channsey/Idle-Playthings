from instapy import InstaPy
from instapy import smart_run
import random
"""
@ToDo
add tags to avoid thirsty smut

Consider specific interactions for select VIP users

Setup different script for friends with dead accounts
Those dead accounts will like all my posts >:D
They'll also comment shit like "u':heart:' Senpai!


"""
# get a session!
session = InstaPy(username='channsey113', password='password', use_firefox=False, headless_browser=False) 
# let's go! :>
with smart_run(session):
    # settings
    session.set_relationship_bounds(enabled=False,
                                    potency_ratio=None,
                                    delimit_by_numbers=False,
                                    max_followers=6666,
                                    max_following=666,
                                    min_followers=66,
                                    min_following=6,
                                    min_posts=6,
                                    max_posts=66666)
    session.set_simulation(enabled=True)
    session.set_skip_users(skip_private=False, private_percentage=100)

    # completely ignore liking images from certain users
    session.set_ignore_users(['fake_friend'])

    # will prevent commenting on and unfollowing your good friends (the images will
    # still be liked)
    # include good friends and tinders
    session.set_dont_include(['best_buddy'])

    session.set_do_like(enabled=True, percentage=100)

    # This is used to check the number of existing likes a post has
    # and if it either exceed the maximum value set
    # OR does not pass the minimum value set then it will not like that post
    session.set_delimit_liking(enabled=True, max=None, min=8)

    """
    words starting with # will match only exact hashtags (e. g. #cat matches #cat, but not #catpic)
    words starting with [ will match all hashtags starting with your word (e. g. [cat matches #catpic, #caturday and so on)
    words starting with ] will match all hashtags ending with your word (e. g. ]cat matches #mycat, #instacat and so on)
    words without these prefixes will match all hashtags that contain your word regardless of position in hashtag
    (e. g. cat will match #cat, #mycat, #caturday, #rainingcatsanddogs and so on)
    """
    session.set_dont_like(['dog','pup'])

    session.set_ignore_if_contains(['dog', 'pup', 'ex', 'bde', 'girlfriend', 'boyfriend', 'butt', 
                                    u':sweat_drops:', u':joy:', u':tongue:', u':kiss:',  u':peach:', u':eggplant:',
                                    u':bikini:',  u':100:', u':eyes:'])
    session.set_action_delays(enabled=True, like=random.randint(31, 53))


    # Activity

    num = 1
    session.set_user_interact(amount= num, randomize=True, percentage=100, media='Photo')
    # hit up my own followers
    # default amount is 10
    session.interact_user_followers([], amount=num, randomize=False)

    session.set_user_interact(amount= num, randomize=True, percentage=100, media='Photo')
    # This is used to perform likes on your own feeds
    # amount=100  specifies how many total likes you want to perform
    # randomize=True randomly skips posts to be liked on your feed
    # unfollow=True unfollows the author of a post which was considered
    # inappropriate interact=True visits the author's profile page of a
    # certain post and likes a given number of his pictures, then returns to feed
    session.like_by_feed(amount=num, randomize=True, unfollow=False, interact=False)

    # previously known to go by interact's amount
    # session.set_user_interact(amount=num, randomize=True, percentage=100, media='Photo')
    # The VIP Lounge
    session.interact_by_users(['best_buddy'], amount=num, randomize=True, media='Photo')
