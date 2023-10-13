package wikiedits;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.cep.CEP;
import org.apache.flink.cep.pattern.Pattern;
import org.apache.flink.cep.PatternSelectFunction;
import org.apache.flink.cep.pattern.conditions.SimpleCondition;
import org.apache.flink.cep.pattern.conditions.IterativeCondition;
import org.apache.flink.cep.PatternStream;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.common.serialization.StringDeserializer;
import java.util.Properties;

import java.util.List;
import java.util.Map;
import java.util.ArrayList;

public class WikipediaAnalysis {

    public static void main(String[] args) throws Exception {
        
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        Properties kafkaProps = new Properties();
        kafkaProps.setProperty(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        kafkaProps.setProperty(ConsumerConfig.GROUP_ID_CONFIG, "flink-group");
        // FlinkKafkaConsumer<String> kafkaConsumer = new FlinkKafkaConsumer<>("events", new SimpleStringSchema(), properties);
        KafkaSource<String> kafkaSource = KafkaSource.<String>builder()
            .setBootstrapServers("localhost:9092")
            .setTopics("events")
            .setStartingOffsets(OffsetsInitializer.latest())
            .setValueOnlyDeserializer(new SimpleStringSchema())
            .build();
        DataStream<String> kafkaStream = env.fromSource(kafkaSource,WatermarkStrategy.noWatermarks(), "Kafka Source");
        // DataStream<Tuple6<Integer, Integer, Double, Double, Double, Double>> dataStream = env.fromElements(new Tuple2<>("A", 1),new Tuple2<>("B", 2),new Tuple2<>("D", 0),new Tuple2<>("C", 3));
        // Pattern<Tuple6<Integer, Integer, Double, Double, Double, Double>, ?> pattern = Pattern.<Tuple6<Integer, Integer, Double, Double, Double, Double>>begin("start")
        //         .where(SimpleCondition.of(value -> value.getElement2()==5));
                // .oneOrMore();
                // .followedBy("end")
                // .where(SimpleCondition.of(value -> value.f1>=2));
                // .where(new IterativeCondition<Tuple6<Integer, Integer, Double, Double, Double, Double>>() {
                // 	@Override
                // 	public boolean filter(Tuple6<Integer, Integer, Double, Double, Double, Double> value, Context<Tuple6<Integer, Integer, Double, Double, Double, Double>> context) throws Exception {
                //     	return true;
                // 	}
            	// });
                // .where(new SimpleCondition<Tuple6<Integer, Integer, Double, Double, Double, Double>>() {
                //     @Override
                //     public boolean filter(Tuple6<Integer, Integer, Double, Double, Double, Double> value) throws Exception {
                //         return value.f1 >= 3;
                //     }
                // });
        Pattern<String, ?> pattern = Pattern.<String>begin("start")
                .where(SimpleCondition.of(value -> true));
        // Apply the CEP pattern to the DataStream
        // DataStream<Map<String,List<Tuple6<Integer, Integer, Double, Double, Double, Double>>>> resultStream = CEP.pattern(dataStream, pattern).inProcessingTime()
        // .select(new PatternSelectFunction<Tuple6<Integer, Integer, Double, Double, Double, Double>, Map<String,List<Tuple6<Integer, Integer, Double, Double, Double, Double>>>>() {
        //     @Override
        //     public Map<String,List<Tuple6<Integer, Integer, Double, Double, Double, Double>>> select(Map<String, List<Tuple6<Integer, Integer, Double, Double, Double, Double>>> pattern) throws Exception {
        //         Map<String, List<Tuple6<Integer, Integer, Double, Double, Double, Double>>> resultList = pattern;
        //         return resultList;
        //     }
        // });
        DataStream<Map<String,List<String>>> resultStream = CEP.pattern(kafkaStream, pattern).inProcessingTime()
        .select(new PatternSelectFunction<String, Map<String,List<String>>>() {
            @Override
            public Map<String,List<String>> select(Map<String, List<String>> pattern) throws Exception {
                Map<String, List<String>> resultList = pattern;
                return resultList;
            }
        });
        // Print the result
        // dataStream.print();
        resultStream.print();

        // Execute the Flink job
        env.execute();
    }
}
