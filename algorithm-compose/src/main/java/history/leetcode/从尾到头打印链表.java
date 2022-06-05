package history.leetcode;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Stack;

public class 从尾到头打印链表 {

    public class ListNode {
        int val;
        ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }
    }

    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        if (listNode==null){
            return null;
        }
        ArrayList<Integer> list = new ArrayList<>();
        while (listNode.next!=null){
            list.add(listNode.val);
        }
//        for(int i = list.size()-1; i >=0 ; i--) {
//            System.out.println(list.indexOf(i));
//        }
        Collections.reverse(list);  //反转链表
        return list;
    }

    public class Solution {
        public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
            Stack<Integer> stack = new Stack<>();
            while (listNode != null) {
                stack.push(listNode.val);
                listNode = listNode.next;
            }

            ArrayList<Integer> list = new ArrayList<>();
            while (!stack.isEmpty()) {
                list.add(stack.pop());
            }
            return list;
        }
    }
}
