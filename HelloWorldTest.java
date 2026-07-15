import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

/**
 * HelloWorld 单元测试
 * TDD RED 阶段：先写测试，验证 HelloWorld 输出 "Hello, World!"
 */
public class HelloWorldTest {
    
    private static int passed = 0;
    private static int failed = 0;
    
    public static void main(String[] args) {
        testOutputsHelloWorld();
        testClassExists();
        
        System.out.println("\n=== 测试结果 ===");
        System.out.println("通过: " + passed + ", 失败: " + failed);
        
        if (failed > 0) {
            System.exit(1);
        }
    }
    
    /**
     * 测试 main 方法输出 "Hello, World!"
     */
    static void testOutputsHelloWorld() {
        // 捕获 System.out
        PrintStream originalOut = System.out;
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream testOut = new PrintStream(baos);
        System.setOut(testOut);
        
        try {
            HelloWorld.main(new String[]{});
            String output = baos.toString().trim();
            
            if ("Hello, World!".equals(output)) {
                passed++;
                System.setOut(originalOut);
                System.out.println("[PASS] testOutputsHelloWorld: 输出正确 -> " + output);
            } else {
                failed++;
                System.setOut(originalOut);
                System.out.println("[FAIL] testOutputsHelloWorld: 期望 'Hello, World!' 实际 '" + output + "'");
            }
        } catch (Exception e) {
            failed++;
            System.setOut(originalOut);
            System.out.println("[FAIL] testOutputsHelloWorld: 异常 -> " + e.getMessage());
        }
    }
    
    /**
     * 测试 HelloWorld 类存在且可加载
     */
    static void testClassExists() {
        try {
            Class<?> clazz = Class.forName("HelloWorld");
            passed++;
            System.out.println("[PASS] testClassExists: HelloWorld 类存在");
        } catch (ClassNotFoundException e) {
            failed++;
            System.out.println("[FAIL] testClassExists: HelloWorld 类未找到");
        }
    }
}