import numpy as np
import pandas as pd


class ProjectEuler:
    """
    A class containing solutions to various projectuler.net problems
    """
    
    def __init__(self):
        pass

    def smallest_multiple(self,start_number:int = 1, end_number:int = 20,step:int = 1):
        """
        Solves project euler problem 5:

        2520 is the smallest number that can be divided by each of the numbers from 1 to 10 without any remainder.
        What is the smallest positive number that is evenly divisible by all of the numbers from 1 to 20?
        This solution was found to be 232792560

        The problem has been modified such that this function will return the smallest positive number that
        is easily divisible by all numbers from start_number to end_number. A step size can be given if a step
        other than one between start_number and end_number is desired. This is a brute force method and I'm absoltely sure
        there are more efficient solutions but they are not occuring to me at present. 

        Args:
            start_number (int): the number which is the beginning of the range of numbers to be applied to this problem. Default is 1.
            end_number (int): the end number which is the stopping point for the range of numbers to be applied ot this problem. Default is 20.
            step (int): step size to take when calculating function. Defaults to 1.
        Returns:
            solution (int): the number found to be evenly divisible by all numbers from start_number to end_number with the appropriate step.
        """

        numbers = list(range(start_number,end_number+1,step)) #get list of numbers to test against
        solution = None
        test_number = end_number #start at the end_number becuase the test_number needs to be divisible by the end_number.
                                 #any number before the end number is obviously not divisible by the end_number
        while solution == None:
            test_list = []
            for num in numbers:
                test_list.append(test_number%num)
            if sum(test_list) == 0: #all numbers should have a remainder of 0 when divided into the answer
                solution = test_number
            else:
                test_number += end_number #increment test_number by end_number, it stands to reason that if the current test_number is not divisible by
                                          #the end_number, nothing will be until the next test_number+end_number
        return solution

    def self_powers(self):
        """
        Solves project euler problem 48:

        The series, 1**1 + 2**2 + 3**3 + ... + 10**10 = 10405071317.
        Find the last ten digits of the series, 1**1 + 2**2 + 3**3 + ... + 1000**1000.
        This solution was found to be 911084670

        Returns:
            int: the solution of the problem
        """
        solution = 0
        for i in range(1000): #the range of the problem
            solution += i**i #simply add each number to the solution
        return int(str(solution)[-10:-1]) #integers are not subscriptable so convert to string to get subscripts then convert back to int

class Rosalind:
    """
    A class containing solutions to problems from rosalind.info/problems
    """

    def __init__(self):
        pass

    def dna_to_rna(self,dna:str):
        """
        Solves Rosalind problem ID: RNA

        An RNA string is a string formed from the alphabet containing 'A', 'C', 'G', and 'U'.
        Given a DNA string t corresponding to a coding strand, its transcribed RNA string u is formed by replacing all occurrences of 'T' in t with 'U' in u.

        Given: A DNA string t having length at most 1000 nt.
        Return: The transcribed RNA string of t.

        Sample Dataset
            GATGGAACTTGACTACGTAAATT
        Sample Output
            GAUGGAACUUGACUACGUAAAUU

        Args:
            string (str): DNA string

        Raises:
            SyntaxError: Raises if DNA string contains unexpected characters. Expects ATGC

        Returns:
            string: The transcribed RNA string
        """
        
        if set(dna) != set("ATGC"):
            raise SyntaxError("DNA string should only contain letters ATGC")
        
        return dna.replace('T','U')
    
    def find_dna_motif(self,dna:str,motif:str):
        """
        Solves Rosalind problem ID: SUBS

        Given two strings s and t, t is a substring of s if t is contained as a contiguous collection of symbols in s (as a result, t must be no longer than s).
        The position of a symbol in a string is the total number of symbols found to its left, including itself 
        (e.g., the positions of all occurrences of 'U' in "AUGCUUCAGAAAGGUCUUACG" are 2, 5, 6, 15, 17, and 18). The symbol at position i of s is denoted by s[i].

        A substring of s can be represented as s[j:k], where j and k represent the starting and ending positions of the substring in s; 
        for example, if s = "AUGCUUCAGAAAGGUCUUACG", then s[2:5] = "UGCU".

        The location of a substring s[j:k] is its beginning position j; 
        note that t will have multiple locations in s if it occurs more than once as a substring of s (see the Sample below).

        Given: Two DNA strings s and t (each of length at most 1 kbp).

        Return: All locations of t as a substring of s.

        Sample Dataset
        dna = GATATATGCATATACTT
        motif = ATAT
        Sample Output
        2 4 10

        Args:
            dna (str): The string of dna to test the motif against
            motif (str): a pattern to find in the dna
        Returns:
            list: A list of the indexes where the start of each motif was found
        """
        indexes = []
        done=False
        idx=0
        while done == False:
            idx = dna.find(motif,idx)
            if idx != -1:
                indexes.append(idx)
            else:
                done=True
            idx+=1
        return indexes
