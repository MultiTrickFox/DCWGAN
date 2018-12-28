import res

def create():
    generator = res.pickle_load('generator.pkl')
    if generator:
        try:
            res.imgmake(generator, hm=5)
        except Exception as e: print('Error generating: ',e)
    else: print('Generator not found.')

if __name__ == '__main__':
    create()