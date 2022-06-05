package src;

public class 删除链表中重复的结点 {
    public class ListNode {
        int val;
        ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }
    }

    public ListNode deleteDuplication(ListNode pHead)
    {

        if (pHead == null) return null;
        ListNode p = pHead;
        ListNode n = new ListNode(0);
        ListNode pre = n;
        n.next = pHead;
        boolean flag = false;
        while (p != null) {
            ListNode q = p.next;
            if (q == null) break;
            if (q.val == p.val) {
                while (q != null && q.val == p.val) {
                    q = q.next;
                }
                pre.next = q;
                p = q;
            } else {
                if (!flag) {
                    n.next = p;
                    flag = true;
                }
                pre = p;
                p = q;
            }
        }
        return n.next;
    }
}
