import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution1DLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

/*
author: farizrahman4u
 */
public class FastTextCNN extends BaseFastTextClassifier {

    public FastTextCNN(){
        super();
    }
    public FastTextCNN(int numClasses, int inputLength){
        super(numClasses, inputLength);
    }
    protected ComputationGraphConfiguration getModelConfig(){
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU)
                .updater(new Adam(0.01))
                .convolutionMode(ConvolutionMode.Same)
                .l2(0.0001)
                .graphBuilder()
                .addInputs("input")
                .addLayer("cnn3",
                        new Convolution1DLayer.Builder()
                                .kernelSize(3)
                                .stride(1)
                                .nOut(10)
                                .build(), "input")
                .addLayer(
                        "cnn4",
                        new Convolution1DLayer.Builder()
                                .kernelSize(4)
                                .stride(1)
                                .nOut(10)
                                .build(), "input"
                )
                .addLayer(
                        "cnn5",
                        new Convolution1DLayer.Builder()
                                .kernelSize(5)
                                .stride(1)
                                .nOut(10)
                                .build(), "input"
                )
                .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")
                .addLayer("pool1",
                        new GlobalPoolingLayer.Builder()
                                .poolingType(PoolingType.MAX)
                                .dropOut(0.5)
                                .build(), "merge")
                .addLayer("out", new OutputLayer.Builder()
                        .lossFunction(new LossMCXENT())
                        .activation(Activation.SOFTMAX)
                        .nOut(getNumClasses()).build(), "pool1")
                .setOutputs("out")
                .setInputTypes(InputType.recurrent(300, getInputLength()))
                .build();

        return config;
    }
}
