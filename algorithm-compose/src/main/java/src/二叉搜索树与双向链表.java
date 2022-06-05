package src;//输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

public class 二叉搜索树与双向链表
{

    public class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;

        public TreeNode(int val) {
            this.val = val;

        }

    }
    public TreeNode Convert(TreeNode root)
    {

        if(root==null)return null;
        if(root.left==null&&root.right==null)return root;
        TreeNode left=Convert(root.left);
        TreeNode p=left;
        while(p!=null&&p.right!=null)
        {
            p=p.right;
        }
        if(left!=null)
        {
            p.right=root;
            root.left=p;
        }
        TreeNode right=Convert(root.right);
        if(right!=null)
        {
            root.right=right;
            right.left=root;
        }

        return  left!=null?left:root;
    }
}