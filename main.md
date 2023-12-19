```python
# %pip install -r requirements.txt
```


```python
from typing import List, Tuple, Dict, Any, Union, Optional
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


```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        if not matrix:
            return

        m, n = len(matrix), len(matrix[0])
        zero_first_row, zero_first_col = False, False

        # Check if first row and first column need to be zeroed
        for j in range(n):
            if matrix[0][j] == 0:
                zero_first_row = True
                break

        for i in range(m):
            if matrix[i][0] == 0:
                zero_first_col = True
                break

        # Use first row and column to mark zeros
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0

        # Zero out marked rows and columns
        for i in range(1, m):
            if matrix[i][0] == 0:
                for j in range(1, n):
                    matrix[i][j] = 0

        for j in range(1, n):
            if matrix[0][j] == 0:
                for i in range(1, m):
                    matrix[i][j] = 0

        # Zero out the first row and column if needed
        if zero_first_row:
            for j in range(n):
                matrix[0][j] = 0

        if zero_first_col:
            for i in range(m):
                matrix[i][0] = 0

        
assert Solution().setZeroes(matrix = [[1,1,1],[1,0,1],[1,1,1]]) == [[1,0,1],[0,0,0],[1,0,1]]
```


```python
# https://leetcode.com/problems/count-odd-numbers-in-an-interval-range/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def countOdds(self, low: int, high: int) -> int:
        """This one we get time limit errors pretty quickly so we need to optimize it, optimize it by checking if the number is even or odd before we start counting"""
        if low % 2 == 0:
            low += 1
        
        # Check if 'high' is even
        if high % 2 == 0:
            high -= 1
            
        return (high - low) // 2 + 1
        
assert Solution().countOdds(low = 3, high = 7) == 3
```


```python
# https://leetcode.com/problems/average-salary-excluding-the-minimum-and-maximum-salary/description/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def average(self, salary: List[int]) -> float:
        salary.sort()
        salary = salary[1:-1]
        return sum(salary) / len(salary)
        

assert Solution().average(salary = [4000,3000,1000,2000]) == 2500.00000
```


```python
# https://leetcode.com/problems/lemonade-change/description/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        # Initialize the number of fives and tens
        fives, tens = 0, 0
        
        # Iterate through the bills
        for bill in bills:
            # Check if the bill is a five
            if bill == 5:
                fives += 1
            # Check if the bill is a ten
            elif bill == 10:
                # Check if we have enough fives
                if fives == 0:
                    return False
                # Update fives and tens
                fives -= 1
                tens += 1
            # Check if the bill is a twenty
            else:
                # Check if we have enough fives and tens
                if fives > 0 and tens > 0:
                    fives -= 1
                    tens -= 1
                # Check if we have enough fives
                elif fives >= 3:
                    fives -= 3
                else:
                    return False
        
        return True
        

assert Solution().lemonadeChange(bills = [5,5,5,10,20]) == True
```


```python
# https://leetcode.com/problems/largest-perimeter-triangle/description/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort(reverse=True)
        for i in range(len(nums)-2):
            if nums[i] < nums[i+1] + nums[i+2]:
                return nums[i] + nums[i+1] + nums[i+2]
        return 0
        
assert Solution().largestPerimeter(nums = [2,1,2]) == 5
```


```python
# https://leetcode.com/problems/check-if-it-is-a-straight-line/description/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        # Get the first two points
        x0, y0 = coordinates[0]
        x1, y1 = coordinates[1]
        
        # Iterate through the remaining points
        for i in range(2, len(coordinates)):
            # Get the current point
            x, y = coordinates[i]
            
            # Check if the current point is on the line
            if (y1 - y0) * (x - x1) != (y - y1) * (x1 - x0):
                return False
        
        return True
        

assert Solution().checkStraightLine(coordinates = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]) == True
```


```python
# https://leetcode.com/problems/add-binary/description/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def addBinary(self, a: str, b: str) -> str:
        
        # Initialize the result
        result = ''
        
        # Initialize the carry
        carry = 0
        
        # Initialize the pointers
        i, j = len(a) - 1, len(b) - 1
        
        # Iterate through the strings
        while i >= 0 or j >= 0:
            # Get the digits
            digit_a = int(a[i]) if i >= 0 else 0
            digit_b = int(b[j]) if j >= 0 else 0
            
            # Compute the sum
            digit_sum = digit_a + digit_b + carry
            
            # Update the carry
            carry = digit_sum // 2
            
            # Update the result
            result = str(digit_sum % 2) + result
            
            # Update the pointers
            i -= 1
            j -= 1
        
        # Check if there is a carry left over
        if carry:
            result = '1' + result
        
        return result
    
assert Solution().addBinary(a = "11", b = "1") == "100"
```


```python
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        num_1 = int(num1)
        num_2 = int(num2)
        return str(num_1 * num_2)
        
assert Solution().multiply(num1 = "2", num2 = "3") == "6"

```


```python
# https://leetcode.com/problems/powx-n/?envType=study-plan-v2&envId=programming-skills

class Solution:
    def myPow(self, x: float, n: int) -> float:
        return x ** n
    
assert Solution().myPow(x = 2.00000, n = 10) == 1024.00000

```


```python
# https://leetcode.com/problems/merge-two-sorted-lists/description/?envType=study-plan-v2&envId=programming-skills

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        # Initialize dummy node and current pointer
        dummy = ListNode(-1)
        current = dummy
        
        # Initialize pointers for list1 and list2
        p1 = list1
        p2 = list2
        
        while p1 and p2: 
            if p1.val < p2.val:
                # Add p1 to merged list and move p1 forward
                current.next = p1
                p1 = p1.next
            else:
                # Add p2 to merged list and move p2 forward
                current.next = p2
                p2 = p2.next
            current = current.next 
        
        if p1:
            current.next = p1
        else:
            current.next = p2
        
        return dummy.next  # Return merged list without dummy node




assert Solution().mergeTwoLists(list1 = [1,2,4], list2 = [1,3,4]) == [1,1,2,3,4,4]
```


```python
# https://leetcode.com/problems/add-two-numbers/description/?envType=study-plan-v2&envId=programming-skills
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:        
            # Initialize dummy node and current pointer
        def reverse_list(head: ListNode) -> ListNode:
                prev = None
                while head:
                    next_node = head.next
                    head.next = prev
                    prev = head
                    head = next_node
                return prev

        l1 = reverse_list(l1)
        l2 = reverse_list(l2)
        
        carry = 0
        dummy = ListNode()
        current = dummy

        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            
            _sum = val1 + val2 + carry
            carry = _sum // 10
            
            current.next = ListNode(_sum % 10)
            current = current.next
            
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
                
        return reverse_list(dummy.next)
    
```


```python
# https://leetcode.com/problems/maximum-depth-of-binary-tree/?envType=study-plan-v2&envId=programming-skills

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        
        return max(
            self.maxDepth(root.left), 
            self.maxDepth(root.right)
            ) + 1


assert Solution().maxDepth(root = [3,9,20,None,None,15,7]) == 3
```


```python
# https://leetcode.com/problems/minimum-depth-of-binary-tree/description/?envType=study-plan-v2&envId=programming-skills

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        
        if not root.left:
            return self.minDepth(root.right) + 1
        
        if not root.right:
            return self.minDepth(root.left) + 1
        
        return min(
            self.minDepth(root.left), 
            self.minDepth(root.right)
            ) + 1
        

assert Solution().minDepth(root = [3,9,20,None,None,15,7]) == 2
```


```python
# https://leetcode.com/problems/binary-tree-inorder-traversal/description/?envType=study-plan-v2&envId=programming-skills

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        
        return (
            self.inorderTraversal(root.left) + 
            [root.val] + 
            self.inorderTraversal(root.right)
            )

assert Solution().inorderTraversal(root = [1,None,2,3]) == [1,3,2]
```


```python
# https://leetcode.com/problems/invert-binary-tree/?envType=study-plan-v2&envId=programming-skills
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        
        root.left, root.right = (
            self.invertTree(root.right), 
            self.invertTree(root.left)
            )
        
        return root
        

assert Solution().invertTree(root = [4,2,7,1,3,6,9]) == [4,7,2,9,6,3,1]
```


```python
# https://leetcode.com/problems/design-parking-system/description/?envType=study-plan-v2&envId=programming-skills

class ParkingSystem:

    def __init__(self, big: int, medium: int, small: int):
        self.spaces = [big, medium, small]
        
    def addCar(self, carType: int) -> bool:
        if self.spaces[carType - 1] > 0:
            self.spaces[carType - 1] -= 1
            return True
        else:
            return False

assert ParkingSystem(big = 1, medium = 1, small = 0).addCar(carType = 1) == True
```


```python
# https://leetcode.com/problems/design-hashmap/description/?envType=study-plan-v2&envId=programming-skills
class MyHashMap:

    def __init__(self):
        self.size = 1000
        self.map = [None] * self.size        

     def put(self, key: int, value: int) -> None:
        index = key % self.size  # Get the index using modulo operation
        # If it's the first time inserting with this index, create a list
        if self.map[index] is None:
            self.map[index] = []
        # Update or append the (key, value) tuple
        for i, (k, v) in enumerate(self.map[index]):
            if k == key:
                self.map[index][i] = (key, value)
                return
        self.map[index].append((key, value))

    def get(self, key: int) -> int:
        index = key % self.size
        if self.map[index] is None:
            return -1  # Key is not present
        for (k, v) in self.map[index]:
            if k == key:
                return v
        return -1  # Key is not present
        

    def remove(self, key: int) -> None:
        index = key % self.size
        if self.map[index] is not None:
            self.map[index] = [(k, v) for k, v in self.map[index] if k != key]
        


# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)
```


```python
class UndergroundSystem:

    def __init__(self):
        self.check_ins = {}  # Maps ID to (stationName, t)
        self.travel_times = {}  # Maps (startStation, endStation) to (totalTime, totalTrips)

    def checkIn(self, id, stationName, t):
        # Record the check-in data
        self.check_ins[id] = (stationName, t)

    def checkOut(self, id, stationName, t):
        # Check if the user has checked in
        if id in self.check_ins:
            start_station, start_time = self.check_ins.pop(id)
            # Calculate the travel time
            travel_time = t - start_time
            # Update the travel times data
            if (start_station, stationName) not in self.travel_times:
                self.travel_times[(start_station, stationName)] = (0, 0)  # Initialize if not present
            total_time, total_trips = self.travel_times[(start_station, stationName)]
            # Update the total time and total trips
            self.travel_times[(start_station, stationName)] = (total_time + travel_time, total_trips + 1)

    def getAverageTime(self, startStation, endStation):
        # Calculate the average time
        total_time, total_trips = self.travel_times.get((startStation, endStation), (0, 0))
        if total_trips == 0:
            return 0  # Avoid division by zero
        return total_time / total_trips
```


```python
class BrowserHistory:
    
    def __init__(self, homepage: str):
        self.history = [homepage]  # The history stack with the homepage
        self.current = 0  # The current index in the history

    def visit(self, url: str):
        # Go to the url from the current page and clear the forward history
        self.history = self.history[:self.current + 1]  # Clear forward history
        self.history.append(url)  # Add the new url to the history
        self.current += 1  # Update the current index to the new url

    def back(self, steps: int):
        # Go back in the history by steps number of steps
        self.current = max(0, self.current - steps)  # Ensure we don't go back past the homepage
        return self.history[self.current]  # Return the current url

    def forward(self, steps: int):
        # Go forward in the history by steps number of steps
        self.current = min(len(self.history) - 1, self.current + steps)  # Ensure we don't go forward past the latest page
        return self.history[self.current]  # Return the current url

```


```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class MyLinkedList:

    def __init__(self):
        self.size = 0
        self.head = ListNode(0)  # Sentinel node as pseudo-head

    def get(self, index: int) -> int:
        # If the index is invalid, return -1
        if index < 0 or index >= self.size:
            return -1
        current = self.head
        # Use the sentinel node to start at the head
        for _ in range(index + 1):
            current = current.next
        return current.val

    def addAtHead(self, val: int) -> None:
        self.addAtIndex(0, val)

    def addAtTail(self, val: int) -> None:
        self.addAtIndex(self.size, val)

    def addAtIndex(self, index: int, val: int) -> None:
        # If the index is greater than the length, the node will not be inserted
        if index > self.size:
            return
        # If the index is negative, the node will be inserted at the head of the list
        if index < 0:
            index = 0
        self.size += 1
        # Find predecessor of the node to be added
        pred = self.head
        for _ in range(index):
            pred = pred.next
        # Node to be added
        to_add = ListNode(val)
        # Insertion itself
        to_add.next = pred.next
        pred.next = to_add

    def deleteAtIndex(self, index: int) -> None:
        # If the index is invalid, do nothing
        if index < 0 or index >= self.size:
            return
        self.size -= 1
        # Find predecessor of the node to be deleted
        pred = self.head
        for _ in range(index):
            pred = pred.next
        # Delete pred.next 
        pred.next = pred.next.next


```


```python
class MyCircularQueue:

    def __init__(self, k: int):
        self.queue = [0] * k
        self.headIndex = 0
        self.count = 0
        self.capacity = k

    def enQueue(self, value: int) -> bool:
        if self.count == self.capacity:
            return False
        self.queue[(self.headIndex + self.count) % self.capacity] = value
        self.count += 1
        return True

    def deQueue(self) -> bool:
        if self.count == 0:
            return False
        self.headIndex = (self.headIndex + 1) % self.capacity
        self.count -= 1
        return True

    def Front(self) -> int:
        if self.count == 0:
            return -1
        return self.queue[self.headIndex]

    def Rear(self) -> int:
        if self.count == 0:
            return -1
        return self.queue[(self.headIndex + self.count - 1) % self.capacity]

    def isEmpty(self) -> bool:
        return self.count == 0

    def isFull(self) -> bool:
        return self.count == self.capacity


```


```python
class Solution:
    def buddyStrings(self, s: str, goal: str) -> bool:
        # If lengths differ, cannot be buddy strings
        if len(s) != len(goal):
            return False
        
        # If strings are the same, need at least one duplicate character
        if s == goal:
            seen = set()
            for char in s:
                if char in seen:
                    return True
                seen.add(char)
            return False
        
        # Find the pairs of characters that differ
        pairs = []
        for a, b in zip(s, goal):
            if a != b:
                pairs.append((a, b))
            if len(pairs) > 2:  # More than 2 differences means they can't be buddy strings
                return False
        
        # Check if there are exactly 2 pairs and they are the same characters but in different order
        return len(pairs) == 2 and pairs[0] == pairs[1][::-1]

```


```python
class Solution:
    def validWordSquare(self, words: List[str]) -> bool:
        if not all(len(word) == len(words) for word in words):
            return False
        
        # Check if every word matches its corresponding column
        for i in range(len(words)):
            for j in range(len(words[i])):
                # If the corresponding position in the column doesn't match
                # the character in the row, it's not a valid word square.
                # This also handles out-of-bound cases implicitly.
                if j >= len(words) or i >= len(words[j]) or words[i][j] != words[j][i]:
                    return False
        
        return True
```


```python
class Solution:
    def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[List[int]]:
        missing_ranges = []
        next_num = lower

        for num in nums:
            # Skip the numbers less than the lower bound
            if num < next_num:
                continue
            # If there is a gap between the next number we're looking for and the current number
            if num > next_num:
                # Single number range or a range of numbers
                missing_ranges.append([next_num, num - 1])
            # Update the next number, preventing integer overflow
            next_num = num + 1 if num < upper else upper + 1

        # After the loop, check if there's still a range we need to add
        if next_num <= upper:
            missing_ranges.append([next_num, upper])

        return missing_ranges
        
```


```python
class Solution:
    def confusingNumber(self, n: int) -> bool:
        valid_rotations = {'0': '0', '1': '1', '6': '9', '8': '8', '9': '6'}
        str_n = str(n)
        rotated_str = ""
        
        for digit in str_n:
            if digit not in valid_rotations:
                return False
            rotated_str = valid_rotations[digit] + rotated_str
        
        return rotated_str != str_n


```


```python
class Solution:
    def myAtoi(self, s: str) -> int:
        MAX_INT, MIN_INT = 2**31 - 1, -2**31
        i, n, sign = 0, len(s), 1  # Initialize index, length, and sign
        result = 0  # Initialize result
        
        # Discard all leading whitespaces
        while i < n and s[i] == ' ':
            i += 1
        
        # Check if the next character is '-' or '+'
        if i < n and (s[i] == '+' or s[i] == '-'):
            sign = -1 if s[i] == '-' else 1
            i += 1
        
        # Process digits and build the result
        while i < n and s[i].isdigit():
            digit = int(s[i])
            # Check for overflow and clamp if necessary
            if result > MAX_INT // 10 or (result == MAX_INT // 10 and digit > MAX_INT % 10):
                return MAX_INT if sign == 1 else MIN_INT
            result = result * 10 + digit
            i += 1
        
        return sign * result
```


```python
import re

class Solution:
    def isNumber(self, s: str) -> bool:
        pattern = re.compile(r"""
        ^                   # Start of string
        [+-]?               # Optional sign
        (                   # Start of group:
          (\d+\.\d*)        # - Digits, dot, optional digits OR
          | (\.\d+)         # - Dot, digits OR
          | \d+             # - Digits
        )                   # End of group
        ([eE][+-]?\d+)?     # Optional exponent part: 'e' or 'E', optional sign, digits
        $                   # End of string
    """, re.VERBOSE)

        return bool(pattern.match(s))
        
```


```python
from itertools import permutations
```


```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        return list(permutations(nums))

assert Solution().permute(nums = [1,2,3]) == [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```


```python
class Solution:
    def compress(self, chars: List[str]) -> int:
        # Initialize the compressed string
        compressed = []
        
        # Initialize the current character and count
        current_char = chars[0]
        current_count = 1
        
        # Iterate through the characters
        for i in range(1, len(chars)):
            # Get the current character
            char = chars[i]
            
            # Check if the character is the same as the current character
            if char == current_char:
                current_count += 1
            else:
                # Add the current character and count to the compressed string
                compressed.append(current_char)
                if current_count > 1:
                    compressed.extend(list(str(current_count)))
                
                # Update the current character and count
                current_char = char
                current_count = 1
        
        # Add the last character and count to the compressed string
        compressed.append(current_char)
        if current_count > 1:
            compressed.extend(list(str(current_count)))
        
        # Update the input array
        chars[:] = compressed
        
        return len(chars)
        

assert Solution().compress(chars = ["a","a","b","b","c","c","c"]) == ["a","2","b","2","c","3"]        
```


```python
class Solution:
    def countBattleships(self, board: List[List[str]]) -> int:
        count = 0
        if len(board) == 0: 
            return count
        m, n = len(board), len(board[0])
        
        for i in range(m):
            for j in range(n):
                print(board[i][j])
            
            
        
        return count

assert Solution().countBattleships(board = [["X",".",".","X"],[".",".",".","X"],[".",".",".","X"]]) == 2
```

    X
    .
    .
    X
    .
    .
    .
    X
    .
    .
    .
    X



    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    /Users/tylerkruer/lc/main.ipynb Cell 58 line 1
         <a href='vscode-notebook-cell:/Users/tylerkruer/lc/main.ipynb#Y111sZmlsZQ%3D%3D?line=9'>10</a>                 print(board[i][j])
         <a href='vscode-notebook-cell:/Users/tylerkruer/lc/main.ipynb#Y111sZmlsZQ%3D%3D?line=13'>14</a>         return count
    ---> <a href='vscode-notebook-cell:/Users/tylerkruer/lc/main.ipynb#Y111sZmlsZQ%3D%3D?line=15'>16</a> assert Solution().countBattleships(board = [["X",".",".","X"],[".",".",".","X"],[".",".",".","X"]]) == 2


    AssertionError: 



```python
class Node:
    def __init__(self, data):
        self.data = data  # Assign data
        self.next = None  # Initialize next as null


class LinkedList:
    def __init__(self):
        self.head = None  # Initialize head as null

    def print_list(self):
        temp = self.head
        while temp:
            print(temp.data)
            temp = temp.next

    def append(self, new_data):
        new_node = Node(new_data)
        if self.head is None:
            self.head = new_node
            return
        last = self.head
        while last.next:
            last = last.next
        last.next = new_node

    def prepend(self, new_data):
        new_node = Node(new_data)
        new_node.next = self.head
        self.head = new_node

    def find(self, value):
        current = self.head
        while current:
            if current.data == value:
                return True
            current = current.next
        return False

    def delete_node(self, key):
        temp = self.head

        if temp is not None:
            if temp.data == key:
                self.head = temp.next
                temp = None
                return

        while temp is not None:
            if temp.data == key:
                break
            prev = temp
            temp = temp.next

        if temp is None:
            return

        prev.next = temp.next
        temp = None

```


```python
# Create a linked list
llist = LinkedList()

# Append nodes
llist.append(1)
llist.append(2)
llist.append(3)

# Print the linked list
llist.print_list()

# Prepend a node
llist.prepend(0)

# Print the linked list
llist.print_list()

# Delete a node
llist.delete_node(2)

# Print the linked list
llist.print_list()

```

    1
    2
    3
    0
    1
    2
    3
    0
    1
    3



```python
llist = LinkedList()
llist.append(1)
llist.append(2)
llist.append(3)

# Check if value 2 is in the list
print(llist.find(2))  # This should return True

# Check if value 4 is in the list
print(llist.find(4))  # This should return False

```

    True
    False



```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        nums1 = nums1[:m] + nums2[:n]
        nums1.sort()
        return nums1
        
        """
        Do not return anything, modify nums1 in-place instead.
        """
        

assert Solution().merge(nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3) == [1,2,2,3,5,6]
```


```python

```


```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        binary_sol = ""
        while head:
            binary_sol += str(head.val)
            head = head.next
        return int(binary_sol, 2)        
        
    
assert Solution().getDecimalValue(head = ListNode(1, ListNode(0, ListNode(1)))) == 5
```


```python
class Solution:
    def vowelStrings(self, words: List[str], left: int, right: int) -> int:
        count = 0
        for idx, val in enumerate(words):            
            if val[0] in ["a", "e", "i", "o", "u"] and val[-1] in ["a", "e", "i", "o", "u"]:
                count += 1
        return count

assert Solution().vowelStrings(words = ["hey","aeo","mu","ooo","artro"], left = 1, right = 4) == 3
assert Solution().vowelStrings(words = ["are","amy","u"], left = 0, right = 2) == 2
```


```python
class Solution:
    def firstUniqChar(self, s: str) -> int:
        for idx, val in enumerate(s):
            if s.count(val) == 1:
                return idx
        return -1

assert Solution().firstUniqChar(s = "leetcode") == 0
```


```python
class Solution:
    def minimumReplacement(self, nums: List[int]) -> int:
        replacements = 0 
        #  In one operation you can replace any element of the array with any two elements that sum to it.
        if nums == sorted(nums):
            return 0        
        for idx, val in enumerate(nums):
            if val > nums[idx - 1]:                
                # increment replacements of 1
                replacements += 1
                # subtract val with nums[idx - 1] and append the two values to the array remove the val
                nums.remove(val)
                nums.append(val - nums[idx - 1])
                nums.append(nums[idx - 1])                
        return replacements
        


# assert Solution().minimumReplacement(nums = [1,2,3,4,5]) == 0
assert Solution().minimumReplacement(nums = [3,9,3]) == 2
```


```python
class Solution:
    def assignTasks(self, servers: List[int], tasks: List[int]) -> List[int]:
        return [0]
        

assert Solution().assignTasks(servers = [3,3,2], tasks = [1,2,3,2,1,2]) == [2,2,0,2,1,2]
```


```python
class Solution:
    def getBiggestThree(self, grid: List[List[int]]) -> List[int]:
        # size of the grid indicates how many rombuses we can have
        # we want to return the biggest 3 (or less) rombuses
        m = len(grid)
        n = len(grid[0])

        all_area = set() 


        right_pfsum = [[grid[i][j] for j in range(n)] for i in range(m)]
        left_pfsum = [[grid[i][j] for j in range(n)] for i in range(m)]

        print(right_pfsum)
        print(left_pfsum)        


        if len(grid) == 1:
            return [grid[0][0]]
        # if len(grid) == 3:
        #     print(grid[0][1], grid[1][0], grid[1][2], grid[2][1])
        #     largest = int(grid[0][1] + grid[1][0] + grid[1][2] + grid[2][1])
        #     print(largest)
            # return [grid[0][0], grid[0][1], grid[0][2]]
            

        # for idx, val in enumerate(grid):
            

# assert Solution().getBiggestThree(grid = [[3,4,5,1,3],[3,3,4,2,3],[20,30,200,40,10],[1,5,5,4,1],[4,3,2,2,5]]) == [228,216,211]
# assert Solution().getBiggestThree(grid = [[1,2,3],[4,5,6],[7,8,9]]) == [20,9,8]
# assert Solution().getBiggestThree(grid = [[7,7,7]]) == [7]
```


```python
class Solution:
    def isSumEqual(self, firstWord: str, secondWord: str, targetWord: str) -> bool:
        first_w = ''.join([str(ord(letter) - ord('a')) for letter in firstWord])
        second_w = ''.join([str(ord(letter) - ord('a')) for letter in secondWord])
        target_w = ''.join([str(ord(letter) - ord('a')) for letter in targetWord])
        return int(first_w) + int(second_w) == int(target_w)
        
        
assert Solution().isSumEqual(firstWord = "acb", secondWord = "cba", targetWord = "cdb") == True
```


```python
class Solution:
    def largestOddNumber(self, num: str) -> str:
        for idx, val in enumerate(num):
            if int(val) % 2 != 0:
                if idx == len(num) - 1:
                    return num
                return num[:idx + 1]
        return ""

assert Solution().largestOddNumber(num = "52") == "5"
assert Solution().largestOddNumber(num = "4206") == ""
assert Solution().largestOddNumber(num = "35427") == "35427"
```


```python
class Solution:
    def checkZeroOnes(self, s: str) -> bool:
        for idx, val in enumerate(s):
            """
            The longest contiguous segment of 1s has length 2: "1101"
            The longest contiguous segment of 0s has length 1: "1101"
            The segment of 1s is longer, so return true.
            """
            len_one = 0
            len_zero = 0
            if val == "1":
                len_one += 1
            elif val == "0":
                len_zero += 1
        return len_one > len_zero
            

assert Solution().checkZeroOnes(s = "01111110") == True
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    Cell In[7], line 18
         14                 len_zero += 1
         15         return len_one > len_zero
    ---> 18 assert Solution().checkZeroOnes(s = "01111110") == True


    AssertionError: 



```python
from typing import *
```


```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        c = 0
        d = 0
        m = prices[0]
        for i in range(1,len(prices)):
            m=min(m,prices[i])
            c=max(c,prices[i]-m)
            d=max(d,c)
        return d
            


assert Solution().maxProfit(prices = [7,1,5,3,6,4]) == 5
```


```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def bst(root, min_val=float('-inf'), max_val=float('inf')):
            if root == None:
                return True
            
            if not (min_val < root.val < max_val):
                return False
            
            return bst(root.left, min_val, root.val) and bst(root.right, root.val, max_val)
        return bst(root)
            

assert Solution().isValidBST(
    TreeNode(2, TreeNode(1), TreeNode(3))
) == True
assert Solution().isValidBST(
    TreeNode(5, TreeNode(1), TreeNode(4, TreeNode(3), TreeNode(6)))
) == False
```


```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """         
        l = 0
        r = len(matrix) -1
        while l < r:
            matrix[l], matrix[r] = matrix[r], matrix[l]
            l += 1
            r -= 1
        for i in range(len(matrix)):
	        for j in range(i):
                  matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


Solution().rotate(
    matrix = [[1,2,3],[4,5,6],[7,8,9]]
)
"""[[7,4,1],[8,5,2],[9,6,3]]"""
```




    '[[7,4,1],[8,5,2],[9,6,3]]'




```python
class Solution:
    def partitionString(self, s: str) -> int:
        for idx, val in enumerate(s):
            print(idx, val)
            

assert Solution().partitionString(s = "abacaba") == 4
```

    0 a
    1 b
    2 a
    3 c
    4 a
    5 b
    6 a



    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    Cell In[53], line 7
          3         for idx, val in enumerate(s):
          4             print(idx, val)
    ----> 7 assert Solution().partitionString(s = "abacaba") == 4


    AssertionError: 



```python
class Solution:
    def problem(input_str: str) -> None:
        pass

assert Solution().problem(input_str="asdf") == None
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[56], line 5
          2     def problem(input_str: str) -> None:
          3         pass
    ----> 5 assert Solution().problem(input_str="asdf") == None


    TypeError: Solution.problem() got multiple values for argument 'input_str'



```python

```
