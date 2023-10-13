package btp;
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

public class VideoAnalysis {

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
        Pattern<String, ?> pattern = Pattern.<String>begin("start")
                .where(SimpleCondition.of(value -> true));
        DataStream<Map<String,List<String>>> resultStream = CEP.pattern(kafkaStream, pattern).inProcessingTime()
        .select(new PatternSelectFunction<String, Map<String,List<String>>>() {
            @Override
            public Map<String,List<String>> select(Map<String, List<String>> pattern) throws Exception {
                Map<String, List<String>> resultList = pattern;
                return resultList;
            }
        });
        resultStream.print();
        env.execute();
    }
}
