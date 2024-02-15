from collections import defaultdict
from typing import List


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def deleteDuplicates(self, head):
        prev_true = False
        sol = None
        fake = ListNode(-1)

        while head.next is not None:
            if head == []:
                break
            if head.next.val == head.val:
                prev_true = True
                head = head.next
                continue
            else:
                if prev_true:
                    prev_true = False
                    head = head.next
                    continue
                else:
                    if not sol:
                        sol = ListNode(head.val)
                        fake.next=sol
                    else:
                        sol.next = ListNode(head.val)
                        sol = sol.next
                    head = head.next
        if not prev_true:
            if not sol:
                sol = ListNode(head.val)
            else:
                sol.next = ListNode(head.val)
        return fake.next


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()

if __name__ == "__main__":
    [1,1,2]
    head = ListNode(1)
    head.next = ListNode(1)
    head.next.next = ListNode(2)
    # head.next.next.next = ListNode(3)
    # head.next.next.next.next = ListNode(4)
    # head.next.next.next.next.next = ListNode(4)
    # head.next.next.next.next.next.next = ListNode(5)
    # head = []
    minStack = Solution().deleteDuplicates(head)
