import sys

def warp_headlines(x):
    if x < 0.645:
        return 4.3 * x
    else:
        return 6.275*x - 1.275

def warp_images(x):
    if x < 0.5:
        return 0
    elif x < 0.597:
        return 3.611 * x
    else:
        return 7.058*x - 2.058

def main():
    if sys.argv[1] == 'headlines':
        warper = warp_headlines
    elif sys.argv[1] == 'images':
        warper = warp_images
    else:
        assert False

    for line in sys.stdin:
        x = float(line.strip())
        print warper(x)

if __name__ == "__main__":
    main()
