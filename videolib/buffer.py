import warnings
from typing import Any, List
import copy


class CircularBuffer:
    '''
    Class defining a circular buffer.
    '''
    def __init__(self, buf_size: int) -> None:
        '''
        Args:
            buf_size: Size of the buffer
        '''
        if buf_size <= 0:
            raise ValueError('Buffer size must be positive')
        self.buf_size: int = buf_size
        self._buf: List = []
        self._back_index: int = -1

    def isempty(self) -> bool:
        '''
        Check if buffer is empty.

        Returns:
            Flag denoting if buffer is empty.
        '''
        return len(self._buf) == 0

    def append(self, item: Any) -> None:
        '''
        Append copy of item into circular buffer.

        Args:
            item: Item to be appended.
        '''
        if self.isempty():
            warnings.warn('Appending to empty buffer. Filling with given value')
            self.fill(item)
        else:
            self._back_index = (self._back_index + 1) % self.buf_size
            self._buf[self._back_index] = copy.copy(item)

    def fill(self, item: Any) -> None:
        '''
        Clear buffer and fill with shallow copies of item.

        Args:
            item: Item to be used to fill.
        '''
        self.clear()
        for i in range(self.buf_size):
            self._buf.append(copy.copy(item))
        self._back_index = self.buf_size - 1

    def clear(self) -> None:
        '''
        Clear circular buffer.
        '''
        self._buf.clear()
        self._back_index = -1

    def check_append(self, item: Any) -> None:
        '''
        If buffer is empty, fill. Else, append.

        Args:
            item: Item to be used to fill/append.
        '''
        if self.isempty():
            self.fill(item)
        else:
            self.append(item)

    def __iter__(self):
        '''
        Initialize iteration index and return iterator.
        '''
        self._iter_index = 0
        return self

    def __next__(self) -> Any:
        '''
        Yield next element, traversing buffer in reverse order.
        '''
        if self._iter_index == self.buf_size:
            raise StopIteration
        item = self._buf[self._back_index - self._iter_index]
        self._iter_index += 1
        return item

    def back(self) -> Any:
        '''
        Get top element of circular buffer.
        '''
        if self.isempty():
            raise IndexError('Empty buffer has no back')
        return self._buf[self._back_index]

    def front(self) -> Any:
        '''
        Get top element of circular buffer.
        '''
        if self.isempty():
            raise IndexError('Empty buffer has no front')
        return self._buf[(self._back_index+1) % self.buf_size]

    def center(self) -> Any:
        '''
        Get center element of circular buffer.
        '''
        if self.isempty():
            raise IndexError('Empty buffer has no center')
        center_index = (self._back_index + self.buf_size//2 + 1) % self.buf_size
        return self._buf[center_index]


    def __getitem__(self, index):
        if self.isempty():
            raise IndexError(f'Empty buffer.')
        if index < 0 or index >= self.buf_size:
            raise IndexError(f'Index must be in the range 0-{self.buf_size-1}')
        return self._buf[self._back_index - index]
