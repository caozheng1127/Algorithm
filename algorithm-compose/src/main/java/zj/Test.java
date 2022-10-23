package zj;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.commons.lang3.StringUtils;

import java.util.*;

public class Test {

    //两数之和
    public int[] twoSum(int[] nums, int target) {
        Arrays.sort(nums);
        int i = 0, j = nums.length - 1;
        while (i < nums.length - 1 && j > 0) {
            if (nums[i] + nums[j] == target) {
                return new int[]{nums[i], nums[j]};
            } else {
                if (nums[i] + nums[j] > target) {
                    j--;
                } else {
                    i++;
                }
            }
        }
        return null;
    }

    public int subString(String s) {
        if (StringUtils.isBlank(s)) {
            return 0;
        }
        if (s.length() == 1) {
            return 1;
        }
        Map<Character, Integer> map = Maps.newHashMap();
        int cnt = 0;
        for (int i = 0, j = 0; j < s.length(); j++) {
            if (map.containsKey(s.charAt(j))) {
                i = map.get(s.charAt(j));
            }
            cnt = Math.max(cnt, j - i + 1);
            map.put(s.charAt(j), j + 1);
        }
        return cnt;
    }

    public class ListNode {
        int val;
        ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }
    }

    //翻转链表
    public ListNode reverseList(ListNode head) {
        ListNode newHead = null;
        while (head != null) {
            ListNode next = head.next;
            head.next = newHead;
            newHead = head;
            head = next;
        }
        return newHead;
    }

    //接雨水
    public int trap(int[] height) {
        int res = 0;
        for (int i = 1; i < height.length - 1; i++) {
            int left = 0;
            int right = 0;
            for (int j = i; j >= 0; j--) {
                left = Math.max(left, height[j]);
            }
            for (int j = i; j < height.length; j++) {
                right = Math.max(right, height[j]);
            }
            res = res + Math.min(left, right) - height[i];
        }
        return res;
    }

    //寻找两个有序数组的中位数
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int n1 = nums1.length;
        int n2 = nums2.length;
        if (n1 > n2)
            return findMedianSortedArrays(nums2, nums1);
        int k = (n1 + n2 + 1) / 2;
        int left = 0;
        int right = n1;
        while (left < right) {
            int m1 = left + (right - left) / 2;
            int m2 = k - m1;
            if (nums1[m1] < nums2[m2 - 1])
                left = m1 + 1;
            else
                right = m1;
        }
        int m1 = left;
        int m2 = k - left;
        int c1 = Math.max(m1 <= 0 ? Integer.MIN_VALUE : nums1[m1 - 1],
                m2 <= 0 ? Integer.MIN_VALUE : nums2[m2 - 1]);
        if ((n1 + n2) % 2 == 1)
            return c1;
        int c2 = Math.min(m1 >= n1 ? Integer.MAX_VALUE : nums1[m1],
                m2 >= n2 ? Integer.MAX_VALUE : nums2[m2]);
        return (c1 + c2) * 0.5;
    }

    //合并两个有序链表
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }
        if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }

    //盛最多水的容器
    public int maxArea(int[] height) {
        int res = 0, l = 0, r = height.length - 1;
        while (l < r) {
            res = Math.max(res, Math.min(height[l], height[r]) * (r - l));
            if (height[r] > height[l]) {
                l++;
            } else {
                r--;
            }
        }
        return res;
    }

    //三数之和
    public static List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> ans = Lists.newArrayList();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0) {
                break;
            }
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int l = i + 1;
            int r = nums.length - 1;
            while (l < r) {
                int sum = nums[i] + nums[l] + nums[r];
                if (sum == 0) {
                    ans.add(Lists.newArrayList(nums[i], nums[l], nums[r]));
                    while (l < r && nums[l] == nums[l + 1]) {
                        l++;
                    }
                    while (l < r && nums[r] == nums[r - 1]) {
                        r--;
                    }
                } else if (sum < 0) {
                    l++;
                } else {
                    r--;
                }
            }
        }
        return ans;
    }

    //最大子序和
    public int maxSubArray(int[] nums) {
        int res = nums[0];
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum = sum + nums[i];
            if (sum > res) {
                res = sum;
            }
            if (sum < 0) {
                sum = 0;
            }
        }
        return res;
    }

    //K个一组翻转链表
    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode tail = head;
        for (int i = 0; i < k; i++) {
            if (tail == null) {
                return head;
            }
            tail = tail.next;
        }
        ListNode newHead = reverse(head, tail);
        head.next = reverseKGroup(tail, k);
        return newHead;
    }

    private ListNode reverse(ListNode head, ListNode tail) {
        ListNode pre = null;
        ListNode next = null;
        while (head != tail) {
            next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        return pre;
    }

    //打家劫舍
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int n = nums.length;
        if (n == 1) {
            return nums[0];
        }
        return Math.max(rob(nums, 0, n - 2), rob(nums, 1, n - 1));
    }

    private int rob(int[] nums, int first, int last) {
        int pre2 = 0, pre1 = 0;
        for (int i = first; i <= last; i++) {
            int cur = Math.max(pre1, pre2 + nums[i]);
            pre2 = pre1;
            pre1 = cur;
        }
        return pre1;
    }

    //快排
    public static void quickSort1(int[] nums, int low, int high) {
        int start = low;
        int end = high;
        int base = nums[low];
        while (low < high) {
            while (low < high && base <= nums[high]) {
                high--;
            }
            nums[low] = nums[high];
            while (low < high && base >= nums[low]) {
                low++;
            }
            nums[high] = nums[low];
            nums[low] = base;
        }
        if (low - start > 1) {
            quickSort1(nums, start, low - 1);
        }
        if (end - high > 1) {
            quickSort1(nums, high + 1, end);
        }
    }

    //买卖股票的最佳时机
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int soFarMin = prices[0];
        int maxProfit = 0;
        for (int i = 1; i < prices.length; i++) {
            soFarMin = Math.min(soFarMin, prices[i]);
            maxProfit = Math.max(maxProfit, prices[i] - soFarMin);
        }
        return maxProfit;
    }

    //分发糖果
    public int candy(int[] ratings) {
        int[] left = new int[ratings.length];
        int[] right = new int[ratings.length];
        Arrays.fill(left, 1);
        Arrays.fill(right, 1);
        for (int i = 1; i < ratings.length; i++) {
            if (ratings[i] > ratings[i - 1]) {
                left[i] = left[i - 1] + 1;
            }
        }
        int count = left[ratings.length - 1];
        for (int i = ratings.length - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1]) {
                right[i] = right[i + 1] + 1;
            }
            count = count + Math.max(left[i], right[i]);
        }
        return count;
    }

    //有效的括号
    public boolean isValid(String s) {
        int n = s.length();//括号的长度
        if (n % 2 == 1) {//括号不是成对的直接返回 false
            return false;
        }
        //把所有对比的括号存入 map，对比时用
        Map<Character, Character> pairs = new HashMap<Character, Character>() {{
            put(')', '(');
            put(']', '[');
            put('}', '{');
        }};
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            if (stack.isEmpty()) {
                stack.push(s.charAt(i));
            } else {
                char c = s.charAt(i);
                Character character = pairs.get(c);
                if (stack.peek() == character) {
                    stack.pop();
                } else {
                    stack.push(c);
                }
            }
        }
        return stack.isEmpty();
    }

    //合并K个排序链表
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists.length == 0) {
            return null;
        }
        return solve(lists, 0, lists.length - 1);
    }

    private ListNode solve(ListNode[] arr, int left, int right) {
        if (left == right) {
            return arr[left];
        }
        int mid = left + (right - left) / 2;
        ListNode lNode = solve(arr, left, mid);
        ListNode rNode = solve(arr, mid + 1, right);
        return mergeTwoLists(lNode, rNode);
    }

    //合并区间
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, Comparator.comparingInt(x -> x[0]));

        LinkedList<int[]> list = new LinkedList<>();
        for (int i = 0; i < intervals.length; i++) {

            if (list.size() == 0 || list.getLast()[1] < intervals[i][0]) {
                list.add(intervals[i]);
            } else {
                list.getLast()[1] = Math.max(list.getLast()[1], intervals[i][1]);
            }
        }
        int[][] res = new int[list.size()][2];
        int index = 0;
        while (!list.isEmpty()) {
            res[index++] = list.removeFirst();
        }
        return res;
    }

    //数组中的第K个最大元素
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>(k);
        for (int i = 0; i < nums.length; i++) {
            if (minHeap.size() != k) {
                minHeap.offer(nums[i]);
            } else if (minHeap.peek() < nums[i]) {
                minHeap.poll();
                minHeap.offer(nums[i]);
            }
        }
        return minHeap.peek();
    }

    //字典序的第K小数字
    public int findKthNumber(int n, int k) {
        //k表示第k小的数字，k--表示已经遍历完一个数
        long cur = 1;
        k--;
        while (k > 0) {
            // 以cur为根的子树节点有nodes个
            int nodes = getTotalNodes(n, cur);
            // 如果个数比k少，那么这个部分都可以直接跳过
            if (k >= nodes) {
                // 跳过全部
                k = k - nodes;
                // 往右移一位
                cur++;
            }
            // 如果数量比k多，那么我们要找的结果就一定是以cur下的子节点
            else {
                // 跳过当前结点
                k = k - 1;
                // 往下走一层
                cur = cur * 10;
            }

        }
        return (int) cur;
    }

    //获取当前节点下的所有子节点
    public int getTotalNodes(int n, long cur) {
        long next = cur + 1;//获取当前节点的兄弟节点
        long totalNodes = 0; //初始化计数器
        while (cur <= n) {
            totalNodes += Math.min(n - cur + 1, next - cur);
            cur = cur * 10;
            next = next * 10;
        }
        return (int) totalNodes;
    }

    //整数反转
    public int reverse(int x) {
        long count = 0;
        while (x != 0) {
            count = count * 10 + x % 10;
            x = x / 10;
        }
        return (int) count;
    }

    //回文数
    public boolean isPalindrome(int x) {
        if (x < 0) {
            return false;
        }
        int cur = 0;
        int num = x;
        while (num != 0) {
            cur = cur * 10 + num % 10;
            num = num / 10;
        }
        return cur == x;
    }

    //全排列
    private List<List<Integer>> res = new LinkedList<>();

    public List<List<Integer>> permute(int[] nums) {
        backtrack(nums, new LinkedList<>());
        return res;
    }

    private void backtrack(int[] nums, LinkedList<Integer> track) {
        if (track.size() == nums.length) {
            res.add(new LinkedList<>(track));
            return;
        }

        for (int num : nums) {
            if (track.contains(num)) {
                continue;
            }
            track.add(num);
            backtrack(nums, track);
            track.removeLast();
        }
    }

    //删除链表的倒数第N个节点
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode slow = dummy, fast = dummy;
        for (int i = 0; i <= n; i++) {
            fast = fast.next;
        }
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return dummy.next;
    }

    //下一个排列
    public void nextPermutation(int[] nums) {
        int n = nums.length;
        int i;
        for (i = n - 2; i >= 0; i--) {
            if (nums[i] < nums[i + 1]) {
                break;
            }
        }
        if (i == -1) {
            Arrays.sort(nums);
        } else {
            int j;
            for (j = n - 1; j > i; j--) {
                if (nums[j] > nums[i]) {
                    break;
                }
            }
            int tmp = nums[j];
            nums[j] = nums[i];
            nums[i] = tmp;
            Arrays.sort(nums, i + 1, n);
        }
    }

    //爬楼梯
    public int climbStairs(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    //编辑距离
    public int minDistance(String word1, String word2) {
        int n = word1.length(), m = word2.length();
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 0; i < n + 1; ++i) dp[i][0] = i;
        for (int i = 0; i < m + 1; ++i) dp[0][i] = i;

        for (int i = 1; i < n + 1; ++i) {
            for (int j = 1; j < m + 1; ++j) {
                dp[i][j] = Math.min(Math.min(dp[i][j - 1] + 1, dp[i - 1][j] + 1),
                        dp[i - 1][j - 1] + ((word1.charAt(i - 1) != word2.charAt(j - 1)) ? 1 : 0));
            }
        }
        return dp[n][m];
    }

    //岛屿数量
    public int numIslands(char[][] grid) {
        int count = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    dfs(grid, i, j);
                    count++;
                }
            }
        }
        return count;
    }

    private void dfs(char[][] grid, int i, int j) {
        if (i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || grid[i][j] == '0') return;
        grid[i][j] = '0';
        dfs(grid, i + 1, j);
        dfs(grid, i, j + 1);
        dfs(grid, i - 1, j);
        dfs(grid, i, j - 1);
    }

    //相交链表
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;
        ListNode ptrA = headA;
        ListNode ptrB = headB;
        while (ptrA != ptrB) {
            ptrA = ptrA != null ? ptrA.next : headB;
            ptrB = ptrB != null ? ptrB.next : headA;
        }
        return ptrA;
    }

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    //层次遍历
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int cur = 1, next = 0;
        List<Integer> list = new ArrayList<>();

        while (!queue.isEmpty()) {
            TreeNode t = queue.poll();
            cur--;
            list.add(t.val);
            if (t.left != null) {
                queue.offer(t.left);
                next++;
            }
            if (t.right != null) {
                queue.offer(t.right);
                next++;
            }
            if (cur == 0) {
                cur = next;
                next = 0;
                res.add(list);
                list = new ArrayList<>();
            }
        }
        return res;
    }

    //零钱兑换
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        for (int i = 1; i <= amount; i++) {
            dp[i] = amount + 1;
            for (int c : coins) {
                if (i >= c) dp[i] = Math.min(dp[i], dp[i - c] + 1);
            }
        }
        return dp[amount] == amount + 1 ? -1 : dp[amount];
    }

    //对称二叉树
    public boolean isSymmetric(TreeNode root) {
        return isMirror(root, root);
    }

    public boolean isMirror(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) return true;
        if (t1 == null || t2 == null) return false;
        return (t1.val == t2.val) && isMirror(t1.right, t2.left) && isMirror(t1.left, t2.right);
    }

    //螺旋矩阵
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<>();
        int row = matrix.length;
        if (row == 0) return res;
        int col = matrix[0].length;
        int k = (Math.min(row, col) + 1) / 2;

        for (int i = 0; i < k; i++) {
            for (int a = i; a < col - i; a++) res.add(matrix[i][a]);
            for (int b = i + 1; b < row - i; b++) res.add(matrix[b][col - 1 - i]);
            for (int c = col - 2 - i; (c >= i) && (row - i - 1 != i); c--) res.add(matrix[row - 1 - i][c]);
            for (int d = row - 2 - i; (d > i) && (col - i - 1 != i); d--) res.add(matrix[d][i]);
        }

        return res;
    }

    //二叉树的最大深度
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    //单词搜索
    private int[] row = {1, 0, -1, 0};
    private int[] col = {0, 1, 0, -1};

    public boolean exist(char[][] board, String word) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (dfs(i, j, board, word, 0)) return true;
            }
        }
        return false;
    }

    private boolean dfs(int x, int y, char[][] board, String word, int index) {
        if (x >= board.length || x < 0 || y >= board[0].length || y < 0) return false;
        if (board[x][y] != word.charAt(index)) return false;
        if (index == word.length() - 1) return true;

        board[x][y] = ' ';
        for (int i = 0; i < 4; i++) {
            if (dfs(x + row[i], y + col[i], board, word, index + 1)) return true;
        }
        board[x][y] = word.charAt(index);
        return false;
    }

    //合并两个有序数组
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int[] res = new int[m + n];
        int i = 0;
        int j = 0;
        for (int k = 0; k < m + n; k++) {
            if (i > m - 1) res[k] = nums2[j++];
            else if (j > n - 1) res[k] = nums1[i++];
            else if (nums1[i] <= nums2[j]) res[k] = nums1[i++];
            else if (nums1[i] > nums2[j]) res[k] = nums2[j++];
        }
        System.arraycopy(res, 0, nums1, 0, m + n);
    }

    //重排链表
    public void reorderList(ListNode head) {
        if (head == null || head.next == null) return;
        ListNode slow = head, fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        slow.next = reverseList(slow.next);

        while (slow.next != null && head != null) {
            ListNode cur = slow.next;
            slow.next = cur.next;
            ListNode t = head.next;
            head.next = cur;
            cur.next = t;
            head = head.next.next;
        }
    }

    //括号生成
    public List<String> generateParenthesis(int n) {
        List<String> ans = new ArrayList<>();
        backtrack(ans, "", 0, 0, n);
        return ans;
    }

    private void backtrack(List<String> ans, String cur, int left, int right, int max) {
        if (cur.length() == max * 2) {
            ans.add(cur);
            return;
        }

        if (left < max) backtrack(ans, cur + "(", left + 1, right, max);
        if (right < left) backtrack(ans, cur + ")", left, right + 1, max);
    }

    //路径总和
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) return false;
        if (root.left == null && root.right == null) {
            return root.val == sum;
        }
        return hasPathSum(root.left, sum - root.val)
                || hasPathSum(root.right, sum - root.val);
    }

    //二叉树中的最大路径和
    private int res1 = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        findMax(root);
        return res1;
    }

    //组合总数
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(candidates); // 排序是为了提前终止搜索
        dfs(candidates, target, 0, new ArrayDeque<>(), res);
        return res;
    }

    private void dfs(int[] arr, int target, int begin, Deque<Integer> path, List<List<Integer>> res) {
        if (target == 0) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = begin; i < arr.length; i++) {
            if (target - arr[i] < 0) break;
            path.addLast(arr[i]);
            dfs(arr, target - arr[i], i, path, res);
            path.removeLast();
        }
    }

    //不同路径
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || j == 0) dp[i][j] = 1;
                else dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    private int findMax(TreeNode root) {
        if (root == null) return 0;
        int left = Math.max(0, findMax(root.left));
        int right = Math.max(0, findMax(root.right));
        res1 = Math.max(res1, left + right + root.val);
        return Math.max(left, right) + root.val;
    }

    //朋友圈
    public int findCircleNum(int[][] M) {
        int[] visited = new int[M.length];
        int count = 0;
        for (int i = 0; i < M.length; i++) {
            if (visited[i] == 0) {
                dfs(M, visited, i);
                count++;
            }
        }
        return count;
    }

    private void dfs(int[][] M, int[] visited, int i) {
        for (int j = 0; j < M.length; j++) {
            if (M[i][j] == 1 && visited[j] == 0) {
                visited[j] = 1;
                dfs(M, visited, j);
            }
        }
    }

    //最长上升子序列
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        int res = 0;
        for (int i = 0; i < nums.length; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) dp[i] = Math.max(dp[i], dp[j] + 1);
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    //最长公共子序列
    public int longestCommonSubsequence(String text1, String text2) {
        int n1 = text1.length(), n2 = text2.length();
        int[][] dp = new int[n1 + 1][n2 + 1];
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n2; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[n1][n2];
    }

    //跳跃游戏
    public boolean canJump(int[] nums) {
        int k = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            k = Math.max(k, i + nums[i]);
            if (i >= k) return false;
        }
        return k >= nums.length - 1;
    }

    //跳跃游戏II
    public int jump(int[] nums) {
        int end = 0;
        int maxPosition = 0;
        int steps = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            maxPosition = Math.max(maxPosition, nums[i] + i);
            if (i == end) {
                end = maxPosition;
                steps++;
            }
        }
        return steps;
    }

    //删除重复链表中的重复元素
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode dummy = new ListNode(-1024);
        dummy.next = head;
        ListNode pre = dummy;

        while (head != null) {
            if (head.val == pre.val) {
                pre.next = head.next;
                head.next = null;
                head = pre.next;
            } else {
                pre = head;
                head = head.next;
            }
        }
        return dummy.next;
    }

    //最小路径和
    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0] == null || grid[0].length == 0) {
            return 0;
        }
        int row = grid.length;
        int col = grid[0].length;
        int[][] dp = new int[row][col];
        dp[0][0] = grid[0][0];

        for (int i = 1; i < row; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for (int j = 1; j < col; j++) {
            dp[0][j] = dp[0][j - 1] + grid[0][j];
        }
        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[row - 1][col - 1];
    }

    //LRU
    // 基于linkedHashMap
    public class LRUCache {
        private LinkedHashMap<Integer, Integer> cache;
        private int capacity;   //容量大小

        public LRUCache(int capacity) {
            cache = new LinkedHashMap<>(capacity);
            this.capacity = capacity;
        }

        public int get(int key) {
            //缓存中不存在此key，直接返回
            if (!cache.containsKey(key)) {
                return -1;
            }
            int res = cache.get(key);
            cache.remove(key);   //先从链表中删除
            cache.put(key, res);  //再把该节点放到链表末尾处
            return res;
        }

        public void put(int key, int value) {
            if (cache.containsKey(key)) {
                cache.remove(key); //已经存在，在当前链表移除
            }
            if (capacity == cache.size()) {
                //cache已满，删除链表头位置
                Set<Integer> keySet = cache.keySet();
                Iterator<Integer> iterator = keySet.iterator();
                cache.remove(iterator.next());
            }
            cache.put(key, value);  //插入到链表末尾
        }
    }
}
