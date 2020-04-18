# _*_coding:utf-8_*_
"""
Author: Lingren Kong
Created Time: 2020/3/11 21:51
"""

import itertools


class CartesianProduct(object):
    """
    Class to Build the Cartesian Product
    """

    def __init__(self):
        """
        Initial without any required parameters
        """
        self._list_of_sets=[]

    def add(self, inputset):
        """add a set for the Cartesian Product

        Parameters
        ----------
        inputset: iterable

        Returns
        -------

        """
        self._list_of_sets.append(inputset)

    def delete(self,loc):
        """Delete one set from the Cartisian Product

        Parameters
        ----------
        loc: int the location of the target

        Returns
        -------

        """
        self._list_of_sets.pop(loc)
        return

    @property
    def list_of_sets(self):
        """

        Returns self._list_of_sets
        -------

        """
        return self._list_of_sets

    def run(self): #计算笛卡尔积
        return itertools.product(*self._list_of_sets)

if __name__ == "__main__":
    c = CartesianProduct()
    c.add({1, 2, 3})
    c.add([2,3])
    c.add((1,2))
    output = c.run()
    for i in output:
        print(i)


