package Frame;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class Frame {

    /**
     * 快速排序的逻辑是，若要对 nums[lo..hi] 进行排序，我们先找一个分界点 p，通过交换元素使得 nums[lo..p-1] 都小于等于 nums[p]，
     * 且 nums[p+1..hi] 都大于 nums[p]，然后递归地去 nums[lo..p-1] 和 nums[p+1..hi] 中寻找新的分界点，最后整个数组就被排序了。
     * 快速排序的代码框架如下：
     *
     * @param nums
     * @param lo
     * @param hi
     */
    void sort(int[] nums, int lo, int hi) {
        /****** 前序遍历位置 ******/
        // 通过交换元素构建分界点 p
        int p = partition(nums, lo, hi);
        /************************/

        sort(nums, lo, p - 1);
        sort(nums, p + 1, hi);
    }

    private int partition(int[] nums, int lo, int hi) {
        return 0;
    }

    /**
     * 再说说归并排序的逻辑，若要对 nums[lo..hi] 进行排序，我们先对 nums[lo..mid] 排序，再对 nums[mid+1..hi] 排序，最后把这两个有序的子数组合并，整个数组就排好序了。
     * 归并排序的代码框架如下：
     *
     * @param nums
     * @param lo
     * @param hi
     */
    // 定义：排序 nums[lo..hi]
    void sort1(int[] nums, int lo, int hi) {
        int mid = (lo + hi) / 2;
        // 排序 nums[lo..mid]
        sort1(nums, lo, mid);
        // 排序 nums[mid+1..hi]
        sort1(nums, mid + 1, hi);

        /****** 后序位置 ******/
        // 合并 nums[lo..mid] 和 nums[mid+1..hi]
        merge(nums, lo, mid, hi);
        /*********************/
    }

    private void merge(int[] nums, int lo, int mid, int hi) {
    }

    /**
     * 二叉树遍历框架
     *
     * @param root
     */
    void traverse(TreeNode root) {
        if (root == null) {
            return;
        }
        // 前序位置
        traverse(root.left);
        // 中序位置
        traverse(root.right);
        // 后序位置
    }

    // 输入一棵二叉树的根节点，层序遍历这棵二叉树
    void levelTraverse(TreeNode root) {
        if (root == null) return;
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);

        // 从上到下遍历二叉树的每一层
        while (!q.isEmpty()) {
            int sz = q.size();
            // 从左到右遍历每一层的每个节点
            for (int i = 0; i < sz; i++) {
                TreeNode cur = q.poll();
                // 将下一层节点放入队列
                if (cur.left != null) {
                    q.offer(cur.left);
                }
                if (cur.right != null) {
                    q.offer(cur.right);
                }
            }
        }
    }

    /**
     * 动态规划
     * <p>
     * # 自顶向下递归的动态规划
     * def dp(状态1, 状态2, ...):
     * for 选择 in 所有可能的选择:
     * # 此时的状态已经因为做了选择而改变
     * result = 求最值(result, dp(状态1, 状态2, ...))
     * return result
     * <p>
     * # 自底向上迭代的动态规划
     * # 初始化 base case
     * dp[0][0][...] = base case
     * # 进行状态转移
     * for 状态1 in 状态1的所有取值：
     * for 状态2 in 状态2的所有取值：
     * for ...
     * dp[状态1][状态2][...] = 求最值(选择1，选择2...)
     */

    int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        // 数组大小为 amount + 1，初始值也为 amount + 1
        Arrays.fill(dp, amount + 1);

        // base case
        dp[0] = 0;
        // 外层 for 循环在遍历所有状态的所有取值
        for (int i = 0; i < dp.length; i++) {
            // 内层 for 循环在求所有选择的最小值
            for (int coin : coins) {
                // 子问题无解，跳过
                if (i - coin < 0) {
                    continue;
                }
                dp[i] = Math.min(dp[i], 1 + dp[i - coin]);
            }
        }
        return (dp[amount] == amount + 1) ? -1 : dp[amount];
    }

    /**
     * for 选择 in 选择列表:
     * # 做选择
     * 将该选择从选择列表移除
     * 路径.add(选择)
     * backtrack(路径, 选择列表)
     * # 撤销选择
     * 路径.remove(选择)
     * 将该选择再加入选择列表
     */

    /**
     * 回溯
     * <p>
     * result = []
     * def backtrack(路径, 选择列表):
     * if 满足结束条件:
     * result.add(路径)
     * return
     * <p>
     * for 选择 in 选择列表:
     * 做选择
     * backtrack(路径, 选择列表)
     * 撤销选择
     */

    List<List<Integer>> res = new LinkedList<>();

    /* 主函数，输入一组不重复的数字，返回它们的全排列 */
    List<List<Integer>> permute(int[] nums) {
        // 记录「路径」
        LinkedList<Integer> track = new LinkedList<>();
        // 「路径」中的元素会被标记为 true，避免重复使用
        boolean[] used = new boolean[nums.length];

        backtrack(nums, track, used);
        return res;
    }

    // 路径：记录在 track 中
// 选择列表：nums 中不存在于 track 的那些元素（used[i] 为 false）
// 结束条件：nums 中的元素全都在 track 中出现
    void backtrack(int[] nums, LinkedList<Integer> track, boolean[] used) {
        // 触发结束条件
        if (track.size() == nums.length) {
            res.add(new LinkedList(track));
            return;
        }

        for (int i = 0; i < nums.length; i++) {
            // 排除不合法的选择
            if (used[i]) {
                // nums[i] 已经在 track 中，跳过
                continue;
            }
            // 做选择
            track.add(nums[i]);
            used[i] = true;
            // 进入下一层决策树
            backtrack(nums, track, used);
            // 取消选择
            track.removeLast();
            used[i] = false;
        }
    }

    // 计算从起点 start 到终点 target 的最近距离
//    int BFS(Node start, Node target) {
//        Queue<Node> q; // 核心数据结构
//        Set<Node> visited; // 避免走回头路
//
//        q.offer(start); // 将起点加入队列
//        visited.add(start);
//        int step = 0; // 记录扩散的步数
//
//        while (q not empty) {
//            int sz = q.size();
//            /* 将当前队列中的所有节点向四周扩散 */
//            for (int i = 0; i < sz; i++) {
//                Node cur = q.poll();
//                /* 划重点：这里判断是否到达终点 */
//                if (cur is target)
//                return step;
//                /* 将 cur 的相邻节点加入队列 */
//                for (Node x : cur.adj()) {
//                    if (x not in visited) {
//                        q.offer(x);
//                        visited.add(x);
//                    }
//                }
//            }
//            /* 划重点：更新步数在这里 */
//            step++;
//        }
//    }

    int minDepth(TreeNode root) {
        if (root == null) return 0;
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        // root 本身就是一层，depth 初始化为 1
        int depth = 1;

        while (!q.isEmpty()) {
            int sz = q.size();
            /* 将当前队列中的所有节点向四周扩散 */
            for (int i = 0; i < sz; i++) {
                TreeNode cur = q.poll();
                /* 判断是否到达终点 */
                if (cur.left == null && cur.right == null)
                    return depth;
                /* 将 cur 的相邻节点加入队列 */
                if (cur.left != null)
                    q.offer(cur.left);
                if (cur.right != null)
                    q.offer(cur.right);
            }
            /* 这里增加步数 */
            depth++;
        }
        return depth;
    }

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

}
