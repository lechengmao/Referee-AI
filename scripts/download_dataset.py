import argparse
from getpass import getpass


def main():
    p = argparse.ArgumentParser(description='Download MVFoul using credentials you obtained from SoccerNet')
    p.add_argument('--destination', required=True)
    p.add_argument('--challenge', action='store_true')
    args = p.parse_args()
    from SoccerNet.Downloader import SoccerNetDownloader
    password = getpass('SoccerNet authorized dataset password: ')
    if not password:
        raise ValueError('Password cannot be empty')
    splits = ['train', 'valid', 'test'] + (['challenge'] if args.challenge else [])
    SoccerNetDownloader(LocalDirectory=args.destination).downloadDataTask(
        task='mvfouls', split=splits, password=password)


if __name__ == '__main__':
    main()
