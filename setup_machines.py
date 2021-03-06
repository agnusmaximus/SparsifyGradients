import sys
import os
import threading
from scp import SCPClient
import Queue
import paramiko as pm

PATH_TO_KEYFILE = "/Users/maxlam/.ssh/scienceteam.pem"
SSH_USERNAME = "ubuntu"
SHARED_DIRECTORY_PATH = "nfs"
NFS_DNS_NAME = "fs-9334e03a.efs.us-west-2.amazonaws.com"

INSTALL_NFS_COMMANDS = [
    "sudo apt-get -y update",
    "sudo apt-get -y install nfs-common",
    "sudo mkdir %s" % SHARED_DIRECTORY_PATH,
    "sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 %s:/ %s" % (NFS_DNS_NAME, SHARED_DIRECTORY_PATH),
    "sudo chmod 777 %s" % SHARED_DIRECTORY_PATH
]

INSTALL_SPARSIFY_REPO_COMMANDS = [
    "cd %s" % SHARED_DIRECTORY_PATH,
    "git clone https://github.com/agnusmaximus/SparsifyGradients.git"
]

CD_TO_SHARED_DIRECTORY = [
    "cd %s" % SHARED_DIRECTORY_PATH,
]

def mpi_hostfile_text(hosts):
    return "\n".join(hosts)

# Create a client to the instance
def connect_client(host):
    client = pm.SSHClient()
    client.set_missing_host_key_policy(pm.AutoAddPolicy())
    client.connect(host, username=SSH_USERNAME, key_filename=PATH_TO_KEYFILE)
    return client

def run_ssh_commands(host, commands):
    done = False
    while not done:
       try:
          print("Instance %s, Running ssh commands:\n%s" % (host, "\n".join(commands)))

          # Always need to exit
          commands.append("exit")

          # Set up ssh client
          client = connect_client(host)

          # Clear the stdout from ssh'ing in
          # For each command perform command and read stdout
          commandstring = "\n".join(commands)
          stdin, stdout, stderr = client.exec_command(commandstring)
          output = stdout.read()

          # Close down
          stdout.close()
          stdin.close()
          client.close()
          done = True
       except Exception as e:
          print("Error: " + str(e))
          done = False
    return output

def run_ssh_commands_parallel(host, commands, q):
    output = run_ssh_commands(host, commands)
    q.put((host, output))

def run_command(hosts, commands, quiet=False):
    threads = []
    q = Queue.Queue()
    for host in hosts:
        t = threading.Thread(target=run_ssh_commands_parallel, args=(host, commands, q))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()

    while not q.empty():
        instance, output = q.get()
        if not quiet:
            print(instance, output)

def transfer_keyfile_to_machines(hosts):
    for host in hosts:
       client = connect_client(host)
       scp = SCPClient(client.get_transport())
       print("SCP %s to ~" % (PATH_TO_KEYFILE))
       scp.put(PATH_TO_KEYFILE, "~")
       scp.close()
       client.close()

if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Usage: python setup_machines.py h1,h2,h3...")
        sys.exit(0)

    machines = sys.argv[1].split(",")

    # Remove keyfiles
    run_command(machines, ["sudo rm -rf ~/%s" % PATH_TO_KEYFILE.split("/")[-1]], quiet=True)

    # Transfer keyfile to machines
    transfer_keyfile_to_machines(machines)

    # De-install nfs
    run_command(machines, ["sudo rm -rf %s" % SHARED_DIRECTORY_PATH], quiet=True)

    # Install NFS commands
    run_command(machines, INSTALL_NFS_COMMANDS, quiet=True)

    # Install Sparsify repo
    run_command([machines[0]], INSTALL_SPARSIFY_REPO_COMMANDS, quiet=True)

    # Install hostfile
    run_command([machines[0]], ["touch hostfile"], quiet=True)
    for host in machines:
        command = list(CD_TO_SHARED_DIRECTORY)
        command.append("echo %s >> hostfile" % host)
        run_command([machines[0]], command, quiet=True)

    # Add each other to hosts
    for i, host in enumerate(machines):
        run_command(machines, ["sudo sh -c 'echo %s >> /etc/hosts'" % host], quiet=True)

    for host in machines:
        print("ssh -i %s ubuntu@%s" % (PATH_TO_KEYFILE, host))

    print("Master setup commands:")
    print("eval `ssh-agent -s` && ssh-add ~/%s" % PATH_TO_KEYFILE.split("/")[-1])
    print("ssh-keyscan %s" % " ".join(machines))
    print("cd %s" % SHARED_DIRECTORY_PATH)
    print("mpiexec -n %d -hostfile ./hostfile python [file]" % (len(machines)))
