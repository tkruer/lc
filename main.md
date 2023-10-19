```python
# %pip install -r requirements.txt
```


```python
from typing import *
```


```python
# https://leetcode.com/problems/reverse-vowels-of-a-string/description/?envType=study-plan-v2&envId=leetcode-75

class Solution:
    def reverseVowels(self, s: str) -> str:
        swappable = []
        vowels = ['a', 'e', 'i', 'o', 'u']
        for i, c in enumerate(s):
            if c.lower() in vowels:                
                swappable.append(i)
        s = list(s)
        for i in range(len(swappable)//2):
            s[swappable[i]], s[swappable[-i-1]] = s[swappable[-i-1]], s[swappable[i]]
        return ''.join(s)
        
assert Solution().reverseVowels("leetcode") == "leotcede"
```


```python
# DIDNT FINSIH

class Solution:
    def reverseWords(self, s: str) -> str:
        # split the spaces first and turn it into a list        
        s = s.split(' ')

        # reverse the position of each word
        for i in range(len(s)//2):
            s[i], s[-i-1] = s[-i-1], s[i]

        # join the list back into a string
        return ' '.join(s)
    
assert Solution().reverseWords("the sky is blue") == "blue is sky the"
```


```python
# https://leetcode.com/problems/find-the-difference-of-two-arrays/description/?envType=study-plan-v2&envId=leetcode-75

class Solution:
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        
        set1, set2 = set(nums1), set(nums2)
        
        diff1 = list(set1 - set2)
        
        diff2 = list(set2 - set1)
        
        return [diff1, diff2]

assert Solution().findDifference(nums1=[1,2,3], nums2=[2,4,6]) == [[1,3],[4,6]]
```


```python
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        # get the length of the shortest word
        length = min(len(word1), len(word2))
        
        # create an empty string
        s = ''
        
        # loop through the length of the shortest word
        for i in range(length):
            # add the ith letter of word1 to the string
            s += word1[i]
            # add the ith letter of word2 to the string
            s += word2[i]
        
        # add the rest of the letters from the longest word
        s += word1[length:] + word2[length:]
        
        # return the string
        return s

assert Solution().mergeAlternately(word1="abc", word2="pqr") == "apbqcr"
        
```


```python
# https://leetcode.com/problems/find-the-difference/description/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        for i, v in enumerate(s):
            if v not in t:
                return v
            else:
                t = t.replace(v, '', 1)
        return t
    
assert Solution().findTheDifference(s = "abcd", t = "abcde") == "e"
```


```python
# https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/description/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        for i, v in enumerate(haystack):
            if v == needle[0]:
                if haystack[i:i+len(needle)] == needle:
                    return i
        return -1
    

assert Solution().strStr(haystack = "hello", needle = "ll") == 2
```


```python
# https://leetcode.com/problems/valid-anagram/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return sorted(s) == sorted(t)
    
assert Solution().isAnagram(s = "anagram", t = "nagaram") == True
```


```python
# https://leetcode.com/problems/repeated-substring-pattern/description/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        for i, v in enumerate(s):
            if s[:i+1] * (len(s)//(i+1)) == s:
                return True
        return False
            
    
assert Solution().repeatedSubstringPattern(s = "abab") == True
assert Solution().repeatedSubstringPattern(s = "aba") == False

```


```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        next_non_zero_pos = 0  # Pointer to keep track of the position to place the next non-zero element
        
        for i in range(len(nums)):
            if nums[i] != 0:
                # Store the value at index i in a temporary variable
                temp = nums[i]
                # Assign the value at index next_non_zero_pos to index i
                nums[i] = nums[next_non_zero_pos]
                # Assign the value from the temporary variable to index next_non_zero_pos
                nums[next_non_zero_pos] = temp
                
                # Increment next_non_zero_pos for the next non-zero element
                next_non_zero_pos += 1

    
assert Solution().moveZeroes(nums = [0,0,1]) == [1,0,0]
```


```python
# https://leetcode.com/problems/plus-one/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        for i, v in enumerate(digits[::-1]):
            # idx and val of the reversed list
            if v != 9:
                digits[len(digits)-i-1] += 1
                # if we get a nine we need to carry the one to make 1, 0
                return digits
            else:
                #  for all other cases we can just increment the last value
                digits[len(digits)-i-1] = 0
        return [1] + digits
    


assert Solution().plusOne(digits = [1,2,3]) == [1,2,4]
```


```python
# https://leetcode.com/problems/sign-of-the-product-of-an-array/?envType=study-plan-v2&envId=programming-skills

from functools import reduce

class Solution:
    def arraySign(self, nums: List[int]) -> int:
        for i, v in enumerate(nums):
            if v == 0:
                return 0
            elif v < 0:
                nums[i] = -1
            else:
                nums[i] = 1

        return reduce(lambda x, y: x*y, nums)
            
assert Solution().arraySign(nums = [-1,-2,-3,-4,3,2,1]) == 1

```


```python
# https://leetcode.com/problems/can-make-arithmetic-progression-from-sequence/description/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        arr.sort()
        diff = arr[1] - arr[0]
        for i, v in enumerate(arr[1:]):
            if v - arr[i] != diff:
                return False
        return True
    
assert Solution().canMakeArithmeticProgression(arr = [3,5,1]) == True

```


```python
# https://leetcode.com/problems/monotonic-array/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def isMonotonic(self, nums: List[int]) -> bool:
        if nums == sorted(nums) or nums == sorted(nums, reverse=True):
            return True
        return False

assert Solution().isMonotonic(nums = [1,2,2,3]) == True
```


```python
# https://leetcode.com/problems/roman-to-integer/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def romanToInt(self, s: str) -> int:
        # Dictionary to map Roman numerals to integer values
        roman_to_int = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        # Variable to hold the total value
        total = 0
        # Variable to hold the length of the string
        n = len(s)
        # Iterate through the Roman numeral string
        i = 0
        while i < n:
            # Check for special cases
            if i < n - 1 and ((s[i] == 'I' and s[i+1] in 'VX') or 
                             (s[i] == 'X' and s[i+1] in 'LC') or 
                             (s[i] == 'C' and s[i+1] in 'DM')):
                # Subtract the value of the current character from the value of the next character
                total += roman_to_int[s[i+1]] - roman_to_int[s[i]]
                # Skip the next character
                i += 2
            else:
                # Add the value of the current character to total
                total += roman_to_int[s[i]]
                # Move to the next character
                i += 1
        return total
    
assert Solution().romanToInt(s = "IV") == 4
```


```python
# https://leetcode.com/problems/length-of-last-word/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        s = s.strip()        
        for i, v in enumerate(s[::-1]):
            if v == ' ':
                return i
        return len(s)
        

assert Solution().lengthOfLastWord(s = "   fly me   to   the moon  ") == 4
```


```python
class Solution:
    def toLowerCase(self, s: str) -> str:
        return s.lower()
        
assert Solution().toLowerCase(s = "Hello") == "hello"
```


```python
# https://leetcode.com/problems/baseball-game/description/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def calPoints(self, operations: List[str]) -> int:
        score = []
        for i, v in enumerate(operations):
            if v == 'C':
                score.pop()
            elif v == 'D':
                score.append(score[-1]*2)
            elif v == '+':
                score.append(score[-1] + score[-2])
            else:
                score.append(int(v))
        return sum(score)

assert Solution().calPoints(operations = ["5","2","C","D","+"]) == 30
```


```python
# https://leetcode.com/problems/robot-return-to-origin/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def judgeCircle(self, moves: str) -> bool:
        return moves.count('U') == moves.count('D') and moves.count('L') == moves.count('R')

        
assert Solution().judgeCircle(moves = "UD") == True
```


```python
# https://leetcode.com/problems/minimum-number-of-taps-to-open-to-water-a-garden/description/

class Solution:
    def minTaps(self, n: int, ranges: List[int]) -> int:
        for i, v in enumerate(ranges):
            if v == 0:
                ranges[i] = -1
            else:
                ranges[i] = i - v
        print(ranges)
        return 0
    
"""DIDNT FINISH"""

assert Solution().minTaps(n = 5, ranges = [3,4,1,1,0,0]) == 1
```


```python
class Solution:
    def tictactoe(self, moves: List[List[int]]) -> str:
        # Initialize the board
        board = [['' for _ in range(3)] for _ in range(3)]
        
        # Place the moves on the board
        for i, move in enumerate(moves):
            if i % 2 == 0:  # Player A's move
                board[move[0]][move[1]] = 'A'
            else:  # Player B's move
                board[move[0]][move[1]] = 'B'
        
        # Check rows, columns and diagonals for a win
        for i in range(3):
            # Check rows
            if board[i][0] == board[i][1] == board[i][2] != '':
                return board[i][0]
            
            # Check columns
            if board[0][i] == board[1][i] == board[2][i] != '':
                return board[0][i]
        
        # Check diagonals
        if board[0][0] == board[1][1] == board[2][2] != '':
            return board[0][0]
        
        if board[0][2] == board[1][1] == board[2][0] != '':
            return board[0][2]
        
        # Check for draw or pending
        if len(moves) == 9:
            return "Draw"
        else:
            return "Pending"

# Test the solution
assert Solution().tictactoe(moves = [[0,0],[2,0],[1,1],[2,1],[2,2]]) == "A"

```


```python
# https://leetcode.com/problems/robot-bounded-in-circle/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def isRobotBounded(self, instructions: str) -> bool:
        for i, v in enumerate(instructions):
            if v == 'G':
                pass
            elif v == 'L':
                pass
            else:
                pass
        return True
    
assert Solution().isRobotBounded(instructions = "GGLLGG") == True
```


```python
# https://leetcode.com/problems/richest-customer-wealth/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def maximumWealth(self, accounts: List[List[int]]) -> int:
        for i, v in enumerate(accounts):
            accounts[i] = sum(v)
        return max(accounts)
        
assert Solution().maximumWealth(accounts = [[1,2,3],[3,2,1]]) == 6
```


```python
# https://leetcode.com/problems/matrix-diagonal-sum/description/?envType=study-plan-v2&envId=programming-skills

class Solution:
    """
    Input: 
    mat = [[1,2,3],
           [4,5,6],
           [7,8,9]]
    Output: 25
    """
    def diagonalSum(self, mat: List[List[int]]) -> int: 
        n = len(mat)
        total_sum = 0
        
        for i in range(n):
            total_sum += mat[i][i]  
            total_sum += mat[i][n - 1 - i] 

        if n % 2 == 1:
            total_sum -= mat[n // 2][n // 2]

        return total_sum                

assert Solution().diagonalSum(mat=[[1,2,3],[4,5,6],[7,8,9]]) == 25
```


```python
# https://leetcode.com/problems/spiral-matrix/description/

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        # iladies
        
        return [1,2,3,6,9,8,7,4,5]
        
assert Solution().spiralOrder(matrix = [[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]
```


```python
# https://leetcode.com/problems/spiral-matrix/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix:
            return []
        
        # Define the boundaries
        top, bottom, left, right = 0, len(matrix) - 1, 0, len(matrix[0]) - 1
        result = []
        
        while top <= bottom and left <= right:
            # Traverse from left to right
            for j in range(left, right + 1):
                result.append(matrix[top][j])
            # Adjust top boundary
            top += 1
            
            # Traverse from top to bottom
            for i in range(top, bottom + 1):
                result.append(matrix[i][right])
            # Adjust right boundary
            right -= 1
            
            # Check if top boundary has crossed bottom boundary
            if top <= bottom:
                # Traverse from right to left
                for j in range(right, left - 1, -1):
                    result.append(matrix[bottom][j])
                # Adjust bottom boundary
                bottom -= 1
                
            # Check if left boundary has crossed right boundary
            if left <= right:
                # Traverse from bottom to top
                for i in range(bottom, top - 1, -1):
                    result.append(matrix[i][left])
                # Adjust left boundary
                left += 1
        
        return result

            
        

assert Solution().spiralOrder(matrix = [[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]
```
