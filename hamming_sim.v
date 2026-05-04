module hamming_sim #(parameter D = 4096) (
    input clk,
    input [D-1:0] a,
    input [D-1:0] b,
    output [11:0] sim
);

    wire [D-1:0] xnor_out;

    assign xnor_out = ~(a ^ b);

    popcount #(D) pc (
        .clk(clk),
        .in(xnor_out),
        .count(sim)
    );

endmodule