import socket
def system_config():
    print('Reading system configuration')

    computer_name = socket.gethostname()
    print('computer name = %s\n' % computer_name)
    if computer_name == 'Okapi': # laptop
        SYSINFO = {'computer_name': computer_name,
                    'system': 'Windows',
                    'display': True}
    elif computer_name.startswith('gandalf'):
        SYSINFO = {'computer_name': computer_name,
                    'system': 'Linux',
                    'display': False}
    elif computer_name.startswith('luthin'):
        SYSINFO = {'computer_name': computer_name,
                   'system': 'Linux',
                   'display': False}
    elif computer_name.startswith('pippin'):
        SYSINFO = {'computer_name': computer_name,
                   'system': 'Linux',
                   'display': False}
    else:
        SYSINFO = None

    return SYSINFO