import edu.umd.cloud9.io.pair.PairOfStrings;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.checkerframework.checker.units.qual.A;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;

import static java.lang.Math.exp;


public class Logistic_regression {
    private static Integer dims = 10;

    public static Float dotProduct(ArrayList<Float> x, ArrayList<Float> y) {
        Float sum = 0.0f;
        for (int i = 0; i < x.size(); i++) {
            sum += x.get(i) * y.get(i);
        }
        return sum;
    }

    public static class LRMapper
            extends Mapper<Object, Text, Text, Text> {

        private ArrayList<Float> weights = new ArrayList<Float>();
        private ArrayList<Float> delta = new ArrayList<>();
        public void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            FileSystem fs = FileSystem.get(conf);
            try {
                FSDataInputStream fsInput =
                        fs.open(new Path("/LR-Output/part-r-00000"));
                InputStreamReader fsReader = new InputStreamReader(fsInput);
                BufferedReader buffReader = new BufferedReader(fsReader);

                String line;
                while ((line = buffReader.readLine()) != null) {
                    String[] counting = line.split("\\s+");
                    for(int i = 0; i < counting.length; i++) {
                        weights.add(Float.parseFloat(counting[i]));
                    }
                }
                buffReader.close();
            } catch (Exception e) {
                for(int i = 0; i < dims; i++) {
                    // get random float
                    Random rand = new Random();
                    float random = rand.nextFloat();
                    weights.add(random);
                }
            }
            assert weights.size() != 0;
        }

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            String[] line = value.toString().split("\\s+");
            int label = Integer.parseInt(line[0]);
            ArrayList<Float> pos = new ArrayList<Float>();
            for (int i = 1; i < line.length; i++) {
                pos.add(Float.parseFloat(line[i]));
            }
            float tmp = -dotProduct(weights, pos) * label;
            tmp = (float) exp(tmp);
            tmp = 1.0f / (1.0f + tmp) - 1.0f;
            tmp *= label;

            if (delta.size() == 0) {
                for (int i = 0; i < pos.size(); i++) {
                    delta.add(0.0f);
                }
            }

            for (int i = 0; i < pos.size(); i++) {
                delta.set(i, delta.get(i) + tmp * pos.get(i));
            }

        }
        public void cleanup(Context context) throws IOException, InterruptedException {
            String output = "";
            for (int i = 0; i < delta.size(); i++) {
                output += delta.get(i) + " ";
            }
            context.write(new Text("1"), new Text(output));
        }
    }

    public static class LRReducer
            extends Reducer<Text, Text, Text, Text> {

        public void reduce(Text key, Iterable<Text> values,
                           Context context
        ) throws IOException, InterruptedException {
            ArrayList<Float> delta = new ArrayList<>();
            for (Text val : values) {
                String[] line = val.toString().split("\\s+");
                for (int i = 0; i < line.length; i++) {
                    if (delta.size() == i) {
                        delta.add(Float.parseFloat(line[i]));
                    } else {
                        delta.set(i, delta.get(i) + Float.parseFloat(line[i]));
                    }
                }
            }
            String output = "";
            for (int i = 0; i < delta.size(); i++) {
                output += delta.get(i) + " ";
            }
            context.write(new Text(" "), new Text(output));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.set("mapreduce.job.reduces", "1");
        URI uri = URI.create("hdfs://s0:9000");
        FileSystem fs = FileSystem.get(uri, conf);
        Path inpath = new Path("/LR-Input");
        Path outpath = new Path("/LR-Output");

        if (fs.exists(outpath)) {
            fs.delete(outpath, true);
        }

        long startTime = System.currentTimeMillis();
        dims = 10;
        int iterations = 10;
        System.out.println("[INFO] Dimensions: " + dims);
        System.out.println("[INFO] Iterations: " + iterations);
        for (int i = 0; i < iterations; i++) {
            Job job = Job.getInstance(conf, "LR Job");
            job.setJarByClass(Logistic_regression.class);
            job.setMapperClass(LRMapper.class);
            job.setReducerClass(LRReducer.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);
            job.setMapOutputKeyClass(Text.class);
            job.setMapOutputValueClass(Text.class);
            FileInputFormat.addInputPath(job, new Path("/LR-Input"));
            FileOutputFormat.setOutputPath(job, new Path("/LR-Output"));
            job.waitForCompletion(true);
        }
        long endTime = System.currentTimeMillis();
        System.out.println("Job took " + (endTime - startTime) / 1000.0 + " seconds to complete.");
        System.exit(0);
    }
}