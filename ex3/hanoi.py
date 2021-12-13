import sys
import re


def create_p(domain_file, pegs, disk_stack_list):
    p_list = []
    for peg in pegs:
        for stacked_disk in disk_stack_list:
            p_list.append(peg + '-' + stacked_disk + ' ')
            domain_file.write(peg + '-' + stacked_disk + ' ')
    return p_list


def create_list_of_disk(disks):
    stacks_list = ['']
    for disk in disks:
        for stacked_disks in stacks_list:
            if stacked_disks == disk:
                break
            stacks_list.append(stacked_disks + disk)
    return stacks_list


def get_propositions(disks, pegs, domain_file):
    """
    :param disks: list of disk names
    :param pegs: list of peg names
    :param domain_file: the domain_file
    :return: list of propositions
    """
    disk_stack_list = create_list_of_disk(disks)
    return create_p(domain_file, pegs, disk_stack_list)


def create_domain_file(domain_file_name, n_, m_):
    disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]
    domain_file = open(domain_file_name, 'w')  # use domain_file.write(str) to write to domain_file
    "*** YOUR CODE HERE ***"
    domain_file.write("Propositions:\n")
    proposition_list = get_propositions(disks, pegs, domain_file)
    create_actions(proposition_list, domain_file)
    domain_file.close()


def initiol_writer(problem_file, disks, pegs):
    """
        :param problem_file: file to wri
        :param disks: disks list
        :param pegs: pegs list
        :return:
        """
    problem_file.write('Initial state: ' + pegs[0] + '-')
    for disk in disks:
        problem_file.write(disk)
    for i in range(1, len(pegs)):
        problem_file.write(' ' + pegs[i] + '-')


def goler_writer(problem_file, disks, pegs):
    """

    :param problem_file: file to wri
    :param disks: disks list
    :param pegs: pegs list
    :return:
    """
    problem_file.write('\nGoal state: ')
    for j in range(len(pegs) - 1):
        problem_file.write(pegs[j] + '- ')
    problem_file.write(pegs[-1] + '-')
    for disk in disks:
        problem_file.write(disk)


def create_problem_file(problem_file_name_, n_, m_):
    disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]
    problem_file = open(problem_file_name_, 'w')  # use problem_file.write(str) to write to problem_file

    "*** YOUR CODE HERE ***"
    initiol_writer(problem_file, disks, pegs)
    goler_writer(problem_file, disks, pegs)
    problem_file.close()


def action_helper(prop1, prop2):
    first_disk = '(?:-)(d_\d+)'
    disk_to_m = re.search(first_disk, prop1)
    if disk_to_m is None:
        return None

    digit_cap = '\d+'
    disk_to_m = disk_to_m.group(1)
    disk_move_index = int((re.search(digit_cap, disk_to_m)).group(0))

    top_disk_of_prop2 = re.search(first_disk, prop2)
    if top_disk_of_prop2 is None:
        return disk_to_m, True
    top_disk_of_prop2 = top_disk_of_prop2.group(1)
    top_disk_prop2_index = int((re.search(digit_cap, top_disk_of_prop2)).group(0))
    if disk_move_index >= top_disk_prop2_index:
        return None
    get_digits = '(?<![0-9.])\d+(?![0-9.])'
    list_of_all_digits_in_prop2 = [int(digit) for digit in re.findall(get_digits, prop2)]
    if disk_move_index in list_of_all_digits_in_prop2:
        return None
    return disk_to_m, False


def create_actions(p_list, domain_file):
    domain_file.write('\nActions:\n')

    for prop1 in p_list:
        for prop2 in p_list:
            index_1 = prop1.find('-')
            index_2 = prop2.find('-')
            if prop1[0:index_1] != prop2[0:index_2]:
                val = action_helper(prop1,prop2)
                if val is None:
                    continue
                index = prop2.find('-')
                new_peg1 = prop1.replace(val[0], '')
                pro1 = prop1.replace(' ', '')
                pro2 = prop2.replace(' ', '')
                if not val[1]:
                    new_disk_stack = val[0] + pro2[index + 1:]
                    new_peg2 = pro2[0:index + 1] + new_disk_stack
                else:
                    new_peg2 = pro2 + val[0]
                write_action(pro1, pro2, domain_file, new_peg1,new_peg2)

def write_action(prop1, prop2, domain_file, new_peg1, new_peg2):
    """
    writes action to file
    :param prop1: proposition we're moving from
    :param prop2: proposition we're moving to as a result of the action
    :param domain_file: to write
    """
    domain_file.write('Name: M' + prop1 + '->' + prop2 + '\n')
    domain_file.write('pre: ' + prop1 + ' ' + prop2 + '\n')
    domain_file.write('add: ' + new_peg1 + ' ' + new_peg2 + '\n')
    domain_file.write('delete: ' + prop1 + ' ' + prop2 + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: hanoi.py n m')
        sys.exit(2)

    n = int(float(sys.argv[1]))  # number of disks
    m = int(float(sys.argv[2]))  # number of pegs

    domain_file_name = 'hanoi_%s_%s_domain.txt' % (n, m)
    problem_file_name = 'hanoi_%s_%s_problem.txt' % (n, m)
    create_problem_file(problem_file_name, n, m)
    create_domain_file(domain_file_name, n, m)
