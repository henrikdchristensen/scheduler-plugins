// logging_helpers.go
// CHECKED
package mypriorityoptimizer

// -------------------------
// msg
// -------------------------

// msg formats a log message by combining the messenger and the message.
func msg(messenger string, message string) string {
	return messenger + ": " + message
}
