from recoapis import XMindsRecoApi


def reset():
    print('Resetting Amazon AWS (all datasets/campaigns/...)')
    name, dataset = 'name', 'dataset'
    XMindsRecoApi(name, dataset).reset(only_self=False)


if __name__ == '__main__':
    reset()
